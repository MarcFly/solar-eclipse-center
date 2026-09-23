"""
Solar Eclipse Processor.
High-precision, 16-bit astronomical image processing pipeline for solar eclipse photography:
- Sub-pixel circle & limb detection (flare/prominence immune)
- Coronal streamer extent estimation
- Sub-pixel circle centering
- Single-pass composite affine transformation
- Multi-point prominence rotation alignment and solar registration
- Bulk multi-threaded orchestrator
"""

from eclipse_processor.core.detector import EclipseCircleDetector
from eclipse_processor.core.corona import EclipseCoronaDetector, CoronaResult
from eclipse_processor.core.centering import EclipseCenterer
from eclipse_processor.core.transformer import EclipseAffineTransformer, TransformationResult
from eclipse_processor.core.rotator import EclipseRotationAligner, RotationResult, AnchorPoint
from eclipse_processor.cli import EclipseOrchestrator, main

__all__ = [
    "EclipseCircleDetector",
    "EclipseCoronaDetector",
    "CoronaResult",
    "EclipseCenterer",
    "EclipseAffineTransformer",
    "TransformationResult",
    "EclipseRotationAligner",
    "RotationResult",
    "AnchorPoint",
    "EclipseOrchestrator",
    "main",
]

__version__ = "0.1.0"
