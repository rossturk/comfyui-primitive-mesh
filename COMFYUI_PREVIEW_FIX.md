# ComfyUI Preview Fix

## Problem
ComfyUI was throwing errors when trying to send preview images:
1. First error: `AttributeError: 'Tensor' object has no attribute 'width'` - ProgressBar was receiving a Tensor instead of PIL Image
2. Second error: `KeyError: 'PREVIEW'` - PIL was trying to save with format "PREVIEW" which doesn't exist

## Root Cause
The ProgressBar's `update_absolute()` method does NOT support sending preview images directly. Preview images in ComfyUI are sent via the **PromptServer WebSocket API**, not through the progress bar.

## Solution
Implemented a proper preview system using two components:

### 1. Backend (Python)
- Import `PromptServer` from `server` module
- Convert preview images to base64-encoded PNG data
- Send via WebSocket: `PromptServer.instance.send_sync("sketchbeast.preview", {...})`
- Keep progress bar updates separate (just for progress tracking)

**Key changes in `node.py`:**
```python
# Import PromptServer
from server import PromptServer

# In callback: send preview via WebSocket
buffered = io.BytesIO()
preview_pil.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

PromptServer.instance.send_sync("sketchbeast.preview", {
    "image": img_str,
    "step": step_num,
    "total": total
})
```

### 2. Frontend (JavaScript)
Created `web/js/sketchbeast_preview.js`:
- Registers a ComfyUI extension
- Listens for `sketchbeast.preview` WebSocket messages
- Displays preview image in a floating container (bottom-right corner)
- Shows progress text
- Auto-hides after completion

**Key changes in `__init__.py`:**
```python
import os

# Export WEB_DIRECTORY so ComfyUI loads our JavaScript
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web", "js")
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
```

## Files Modified
1. `comfyui_sketchbeast/node.py` - Added PromptServer import and preview sending logic
2. `comfyui_sketchbeast/__init__.py` - Added WEB_DIRECTORY export
3. `comfyui_sketchbeast/web/js/sketchbeast_preview.js` - Created JavaScript extension

## How It Works
1. During optimization, every N steps (20 previews total):
   - Backend converts current canvas to PIL Image
   - Encodes as base64 PNG
   - Sends via PromptServer WebSocket
2. Frontend JavaScript:
   - Receives WebSocket message
   - Decodes base64 to image
   - Displays in floating preview container
   - Updates progress text

## Testing
After deploying to ComfyUI:
1. Restart ComfyUI server (to load new JavaScript extension)
2. Create workflow with Sketchbeast node
3. Run workflow - should see preview images appear bottom-right
4. Preview updates ~20 times during generation
5. Preview hides 2 seconds after completion

## References
- ComfyUI Custom Node Walkthrough: https://docs.comfy.org/custom-nodes/walkthrough
- ComfyUI Server Communication: https://docs.comfy.org/essentials/comms_overview
- Similar implementation: KSampler uses `latent_preview.py` for intermediate previews
