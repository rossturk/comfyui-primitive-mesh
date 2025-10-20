# Development Notes

## Virtual Environment

To activate the comfy-cli virtual environment for development:

```bash
source /home/rturk/.local/share/pipx/venvs/comfy-cli/bin/activate
```

This venv contains all the necessary dependencies including torch, PIL, numpy, etc.

## Performance Optimizations

### Preview Rendering Fix

**Problem**: The algorithm was taking longer with larger input images, even though optimization happens on a fixed 256x256 canvas.

**Root Cause**: The preview callback was upscaling the 256x256 optimization canvas to the full original resolution (e.g., 4096x4096) every few steps, then PNG encoding and base64 encoding it. For a 4K image with 20 previews, this meant:
- Upscaling 256x256 → 4096x4096 (16M pixels) 20 times
- PNG compressing 16M pixels 20 times
- Sending several MB over WebSocket repeatedly

**Solution**: Send previews using the optimization canvas directly (256x256) without upscaling. The final output is still rendered at full resolution - only the intermediate previews are shown at the optimization resolution.

**Impact**: Dramatic performance improvement for large images, with optimization time now independent of input image size.
