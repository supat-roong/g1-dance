"""Assemble assets/g1/g1_inspire.xml from Unitree's official Inspire-hand URDF.

MuJoCo's URDF importer leaves four gaps (see spec section 7.3), all closed here:

  1. no actuators at all (nu=0)
  2. the 12 <mimic> couplings are silently dropped, so the 24 hand joints come
     in independent; restored as equality constraints with polycoef
  3. the root link is fixed to world, which absorbs the pelvis body and loses
     its 3.813 kg inertia; rebuilt explicitly with a free joint
  4. meshdir="meshes" combines with mesh filenames already prefixed "meshes/"
     to give "meshes/meshes/"
  5. armature is dropped, arriving as 0 on every DoF. The Inspire finger links
     weigh about 1.9 g, against 20 g for the Dex3-1 links in the donor model, so
     a position servo on them is wildly unstable at a 5 ms timestep and blows up
     within a few seconds. Body armature is copied from the donor; hand joints
     get a small armature and damping sized for a 2 g link.

Body PD gains are lifted from the validated g1.xml in g1-manipulation-challenge
so the 29 body DoF behave identically to the model those policies were tuned on.
"""

import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "g1"
URDF = ASSETS / "g1_inspire_dfq.urdf"
OUT = ASSETS / "g1_inspire.xml"
DONOR = Path("/Users/supat/workspace/project/g1-manipulation-challenge/g1.xml")

# pelvis inertial, read from the URDF root link (lost by the importer)
PELVIS = dict(pos="0 0 -0.07605", mass="3.813",
              fullinertia="0.010549 0.0093089 0.0079184 0 2.1e-06 0")
STAND_HEIGHT = 0.793
HAND_KP, HAND_KV = 2.0, 0.1

#: Leg and waist gains, overriding the donor's.
#:
#: The donor model ships ankle kp = 28.5 Nm/rad. Standing needs 10.6 Nm at the
#: ankle, because the default stance puts the CoM 3.2 cm ahead of the feet
#: midpoint, and a position servo can only produce that by sitting at 0.37 rad
#: of tracking error. The ankles therefore sag 21 degrees under static load and
#: pitch the robot over in 1.35 s.
#:
#: Raising them does not buy passive stability and is not meant to. Under pure
#: joint PD the robot is an inverted pendulum on an ankle spring, so standing
#: passively needs stiffness above m*g*h, about 197 Nm/rad for 33.4 kg with the
#: CoM 0.6 m up; 4x the donor ankle reaches 228 and still only survives 1.9 s.
#: Balance is the policy's job. These are the values unitree_rl_gym uses for the
#: G1, chosen so the policy has the authority that stack assumes rather than to
#: make the robot stand on its own.
LEG_GAINS = {
    "hip_yaw": (100.0, 2.0), "hip_roll": (100.0, 2.0), "hip_pitch": (100.0, 2.0),
    "knee": (150.0, 4.0), "ankle_pitch": (40.0, 2.0), "ankle_roll": (40.0, 2.0),
}
# stability floor for a ~2 g finger link at dt=5 ms: kp*dt^2 is 5e-5, so an
# armature of 2e-3 leaves roughly 40x margin without making the finger sluggish
HAND_ARMATURE, HAND_DAMPING = 2e-3, 0.05


def body_gains() -> dict[str, tuple[str, str]]:
    """kp/kv per body joint, from the donor model."""
    if not DONOR.exists():
        return {}
    root = ET.parse(DONOR).getroot()
    out = {}
    for p in root.findall("./actuator/position"):
        j = p.get("joint")
        if j and p.get("kp"):
            out[j] = (p.get("kp"), p.get("kv", "0"))
    return out


def body_armature() -> dict[str, str]:
    """Per-joint armature from the donor model; URDF import loses it (gap 5)."""
    if not DONOR.exists():
        return {}
    root = ET.parse(DONOR).getroot()
    out = {}
    for j in root.iter("joint"):
        if j.get("name") and j.get("armature"):
            out[j.get("name")] = j.get("armature")
    return out


def mimics() -> list[tuple[str, str, float]]:
    """(child, parent, multiplier) from the URDF's <mimic> tags."""
    root = ET.parse(URDF).getroot()
    out = []
    for j in root.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            out.append((j.get("name"), m.get("joint"), float(m.get("multiplier", 1))))
    return out


def urdf_to_mjcf(tmp: Path) -> Path:
    """Compile the URDF and dump MuJoCo's own MJCF for it."""
    src = tmp / "model.urdf"
    text = URDF.read_text().replace('meshdir="meshes"', 'meshdir="."')  # gap 4
    src.write_text(text)
    shutil.copytree(ASSETS / "meshes", tmp / "meshes")
    model = mujoco.MjModel.from_xml_path(str(src))
    out = tmp / "model.xml"
    mujoco.mj_saveLastXML(str(out), model)
    return out


def hand_joints(root: ET.Element) -> list[str]:
    pat = re.compile(r"(thumb|index|middle|ring|pinky)", re.I)
    return [j.get("name") for j in root.iter("joint")
            if j.get("name") and pat.search(j.get("name"))]


def build() -> Path:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tree = ET.parse(urdf_to_mjcf(tmp))
    root = tree.getroot()
    root.set("model", "g1_inspire")

    # mj_saveLastXML writes mesh paths relative to the temp build dir; the
    # shipped model resolves them through meshdir, so reduce to basenames
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f:
            mesh.set("file", f.rsplit("/", 1)[-1])

    for tag, attrs in (
        ("compiler", dict(angle="radian", meshdir="meshes", autolimits="true")),
        ("option", dict(integrator="implicitfast", timestep="0.005",
                        impratio="1.0", cone="pyramidal")),
    ):
        el = root.find(tag)
        if el is None:
            el = ET.SubElement(root, tag)
        for k, v in attrs.items():
            el.set(k, v)

    # gap 3: rebuild the pelvis as a floating-base body owning the old world geoms
    wb = root.find("worldbody")
    pelvis = ET.Element("body", {"name": "pelvis", "pos": f"0 0 {STAND_HEIGHT}"})
    ET.SubElement(pelvis, "freejoint", {"name": "floating_base_joint"})
    ET.SubElement(pelvis, "inertial", PELVIS)
    for child in list(wb):
        wb.remove(child)
        pelvis.append(child)
    wb.append(pelvis)

    # gap 5: restore armature and damping
    arm = body_armature()
    hand_all = set(hand_joints(root))
    for j in root.iter("joint"):
        name = j.get("name")
        if not name:
            continue
        if name in hand_all:
            j.set("armature", str(HAND_ARMATURE))
            j.set("damping", str(HAND_DAMPING))
        elif name in arm:
            j.set("armature", arm[name])

    body_j = [j.get("name") for j in root.iter("joint")
              if j.get("name") and j.get("name") not in set(hand_joints(root))]
    driven = {c for c, _, _ in mimics()}
    hand_independent = [j for j in hand_joints(root) if j not in driven]

    # gap 1: actuators
    for old in root.findall("actuator"):
        root.remove(old)
    act = ET.SubElement(root, "actuator")
    gains = body_gains()
    for j in body_j:
        kp, kv = gains.get(j, ("60", "3"))
        for key, (nkp, nkv) in LEG_GAINS.items():
            if key in j:
                kp, kv = str(nkp), str(nkv)
                break
        ET.SubElement(act, "position",
                      {"name": j, "joint": j, "kp": kp, "kv": kv, "inheritrange": "1"})
    for j in hand_independent:
        ET.SubElement(act, "position",
                      {"name": j, "joint": j, "kp": str(HAND_KP), "kv": str(HAND_KV),
                       "inheritrange": "1"})

    # gap 2: mimic couplings as equality constraints
    for old in root.findall("equality"):
        root.remove(old)
    eq = ET.SubElement(root, "equality")
    for child, parent, mult in mimics():
        ET.SubElement(eq, "joint", {"joint1": child, "joint2": parent,
                                    "polycoef": f"0 {mult} 0 0 0"})

    ET.indent(tree, "  ")
    tree.write(OUT, encoding="unicode")

    scene = ET.Element("mujoco", {"model": "g1_inspire_scene"})
    ET.SubElement(scene, "include", {"file": "g1_inspire.xml"})
    # the renders are a deliverable, so the scene is lit to be readable rather
    # than left on MuJoCo's defaults
    vis = ET.SubElement(scene, "visual")
    ET.SubElement(vis, "headlight", {"ambient": "0.45 0.45 0.45",
                                     "diffuse": "0.7 0.7 0.7",
                                     "specular": "0.2 0.2 0.2"})
    ET.SubElement(vis, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(vis, "global", {"azimuth": "140", "elevation": "-18"})
    asset = ET.SubElement(scene, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
                                     "rgb1": "0.24 0.31 0.42",
                                     "rgb2": "0.06 0.08 0.11",
                                     "width": "512", "height": "1024"})
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d",
                                     "builtin": "checker", "rgb1": "0.28 0.30 0.34",
                                     "rgb2": "0.22 0.24 0.28",
                                     "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid",
                                      "texrepeat": "8 8", "reflectance": "0.08"})
    swb = ET.SubElement(scene, "worldbody")
    ET.SubElement(swb, "geom", {"name": "floor", "type": "plane",
                                "size": "20 20 0.1", "condim": "3",
                                "material": "grid"})
    for x, y in ((2.0, 2.0), (-2.0, 2.0), (0.0, -3.0)):
        ET.SubElement(swb, "light", {"pos": f"{x} {y} 3.5", "dir": f"{-x/4} {-y/4} -1",
                                     "directional": "false", "diffuse": "0.35 0.35 0.35"})
    st = ET.ElementTree(scene)
    ET.indent(st, "  ")
    st.write(ASSETS / "scene.xml", encoding="unicode")
    return OUT


if __name__ == "__main__":
    out = build()
    m = mujoco.MjModel.from_xml_path(str(ASSETS / "scene.xml"))
    print(f"built {out.relative_to(ROOT)}")
    print(f"  nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt} neq={m.neq} ngeom={m.ngeom}")
    assert m.nu == 41, f"expected 41 actuators, got {m.nu}"
    assert m.neq == 12, f"expected 12 mimic equalities, got {m.neq}"
    print("  OK: 29 body + 12 hand actuators, 12 mimic equalities")
