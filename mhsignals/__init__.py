"""
MH-SIGNALS: Mental Health Signal Detection and Response Generation.

End-to-end pipeline: classify intent/concern -> retrieve KB snippets -> generate response.
"""

__version__ = "1.0.0"

from .pipeline import MHSignalsPipeline, Response
