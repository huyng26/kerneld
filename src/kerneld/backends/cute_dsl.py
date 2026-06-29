from __future__ import annotations


class CuteDslBackend:
    name = "cute_dsl"
    build_required = True

    def render_candidate(self, *args, **kwargs):
        raise NotImplementedError("CuTe DSL backend is reserved for future work")

    def build_candidate(self, *args, **kwargs):
        raise NotImplementedError("CuTe DSL backend is reserved for future work")

    def load_entrypoint(self, *args, **kwargs):
        raise NotImplementedError("CuTe DSL backend is reserved for future work")
