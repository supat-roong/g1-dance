"""The platform: pick a component per stage, run it, record what happened.

Each stage of this repo was already a command reading and writing a file, which
is what makes swapping one cheap. What was missing was a way to say which
components a result came from, and a way to compare two results without
re-running the half of the pipeline that did not change. That is all this
package is: a selection, a record, and a cache key.

There is deliberately no score and no ranking. Two standing results in this
repository argue against one: section 17.13's finding that joint error is not
mimicry, and 17.14's that reference fidelity and end-to-end fidelity move in
opposite directions past a point. A composite number would hide both.
"""

from __future__ import annotations

from .record import build_record, compare, read_record, write_record
from .run import run_pipeline
from .spec import STAGES, PipelineSpec, SpecError

__all__ = ["PipelineSpec", "STAGES", "SpecError", "build_record", "compare",
           "read_record", "run_pipeline", "write_record"]
