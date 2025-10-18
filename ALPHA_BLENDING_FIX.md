# Alpha Blending Fix

## Problem
Preview images were showing only one shape repeatedly, even though multiple shapes should have been accumulating on the canvas. On step 12, you'd only see one shape instead of 12 layered shapes.

## Root Cause
The `_draw_step()` method in `step.py` was not properly alpha-blending shapes onto the canvas:

### Old Broken Code:
```python
color_with_alpha = self.color + (int(self.alpha * 255),)  # Created but never used!
self.shape.color = self.color  # Set RGB only, no alpha
self.shape.render(draw)  # Drew directly on image - no alpha blending
```

This caused:
- Shapes to be drawn with RGB color only (no transparency)
- No proper alpha compositing
- Each shape essentially overwrote previous work instead of layering

## Solution
Implemented proper alpha blending using PIL's `alpha_composite`:

### New Fixed Code:
```python
# Convert canvas to RGBA mode
canvas_rgba = ...  # Add alpha channel if needed
img = Image.fromarray(canvas_rgba, mode='RGBA')

# Create transparent overlay for the shape
overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# Draw shape with full RGBA color
color_with_alpha = self.color + (int(self.alpha * 255),)
self.shape.color = color_with_alpha
self.shape.render(draw)

# Composite overlay onto image using proper alpha blending
img = Image.alpha_composite(img, overlay)

# Return RGB for consistency
return result[:, :, :3]
```

## How It Works Now
1. Each shape is drawn on a transparent overlay layer
2. The overlay is alpha-composited onto the accumulated canvas
3. This creates proper transparency blending
4. Shapes accumulate correctly over iterations
5. Preview shows progressive build-up of shapes

## Expected Behavior
- Step 1: See 1 shape
- Step 12: See 12 overlapping shapes with proper transparency
- Step 50: See all 50 shapes blended together
- Final result: Complete artistic rendering with all shapes visible

## Files Modified
- `comfyui_sketchbeast/step.py` - Fixed `_draw_step()` method to use alpha compositing
