"""
ComfyUI Sketchbeast Node
Image-to-vector art generator using geometric shape optimization.

Based on the Sketchbeast algorithm (derivative of primitive.js by Ondřej Žára).
"""

import os
from .node import SketchbeastNode

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SketchbeastNode": SketchbeastNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SketchbeastNode": "Sketchbeast Image to Vector Art"
}

# Web directory for JavaScript extensions
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web", "js")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
