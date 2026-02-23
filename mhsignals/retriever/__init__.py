# Lazy imports so scripts that only need builder don't load search (and faiss)
def __getattr__(name):
    if name == "KBRetriever":
        from .search import KBRetriever
        return KBRetriever
    if name == "KBBuilder":
        from .builder import KBBuilder
        return KBBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
