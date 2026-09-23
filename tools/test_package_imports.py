from eclipse_processor import (
    EclipseCircleDetector,
    EclipseCoronaDetector,
    EclipseCenterer,
    EclipseAffineTransformer,
    EclipseRotationAligner,
    EclipseOrchestrator,
    main,
    __version__
)

print(f"eclipse_processor v{__version__} successfully imported!")
print("Classes successfully loaded:")
print(f"  {EclipseCircleDetector.__name__}")
print(f"  {EclipseCoronaDetector.__name__}")
print(f"  {EclipseCenterer.__name__}")
print(f"  {EclipseAffineTransformer.__name__}")
print(f"  {EclipseRotationAligner.__name__}")
print(f"  {EclipseOrchestrator.__name__}")
