try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
import sys
import os
sam3d_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SAM-Med3D"))
sys.path.append(sam3d_path)

from ._inference_3d import InferQWidget3D
from ._inference_2d import InferQWidget2D
from ._crop_2d import CropQWidget2D
from ._inference_3d_block import  InferQWidget3DBlock
from ._sam_3d import SAMQWidget3D
from ._sam_2d import SAMQWidget2D

from ._widget import (
    ExampleQWidget,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)
__all__ = (
    "ExampleQWidget",
    "ImageThreshold",
    "threshold_autogenerate_widget",
    "threshold_magic_widget",
    "InferQWidget3D",
    "InferQWidget2D",
    "SAMQWidget3D",
    "SAMQWidget2D",
    "CropQWidget2D",
    "InferQWidget3DBlock"
)
