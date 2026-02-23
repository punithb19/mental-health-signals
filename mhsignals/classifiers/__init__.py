# Lazy imports so lightweight scripts don't load heavy classifier deps
def __getattr__(name):
    if name == "BaseClassifier":
        from .base import BaseClassifier
        return BaseClassifier
    if name in ("MinilmLRIntentClassifier", "LoRAIntentClassifier"):
        from .intent import MinilmLRIntentClassifier, LoRAIntentClassifier
        return MinilmLRIntentClassifier if name == "MinilmLRIntentClassifier" else LoRAIntentClassifier
    if name in ("MinilmLRConcernClassifier", "LoRAConcernClassifier"):
        from .concern import MinilmLRConcernClassifier, LoRAConcernClassifier
        return MinilmLRConcernClassifier if name == "MinilmLRConcernClassifier" else LoRAConcernClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
