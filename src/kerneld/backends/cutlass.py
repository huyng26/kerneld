from __future__ import annotations


class CutlassBackend:
    name = "cutlass"
    build_required = True

    def render_candidate(self, *args, **kwargs):
        raise NotImplementedError("CUTLASS backend is reserved for future work")

    def build_candidate(self, *args, **kwargs):
        raise NotImplementedError("CUTLASS backend is reserved for future work")

    def load_entrypoint(self, *args, **kwargs):
        raise NotImplementedError("CUTLASS backend is reserved for future work")
