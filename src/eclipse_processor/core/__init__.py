"""
Core algorithmic modules for solar eclipse image processing:
- detector: Circular solar/lunar limb detector with flare immunity
- corona: Coronal streamer extent detector with dynamic range thresholding
- centering: Sub-pixel translation to canvas center
- transformer: Single-pass composite affine transformation (centering, scaling, rotation, cropping)
- rotator: Multi-point prominence rotation alignment and solar registration
"""

from eclipse_processor.core.detector import EclipseCircleDetector
from eclipse_processor.core.corona import EclipseCoronaDetector, CoronaResult
from eclipse_processor.core.centering import EclipseCenterer
from eclipse_processor.core.transformer import EclipseAffineTransformer, TransformationResult
from eclipse_processor.core.rotator import EclipseRotationAligner, RotationResult, AnchorPoint

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
]
