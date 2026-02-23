"""
MH-SIGNALS: Mental Health Signal Detection and Response Generation.

End-to-end pipeline: classify intent/concern -> retrieve KB snippets -> generate response.
"""

__version__ = "1.0.0"

# Lazy import so scripts that only need config/builder don't pull in heavy deps (e.g. sentence_transformers)
def __getattr__(name):
    if name == "MHSignalsPipeline":
        from .pipeline import MHSignalsPipeline
        return MHSignalsPipeline
    if name == "Response":
        from .pipeline import Response
        return Response
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
