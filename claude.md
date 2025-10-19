# ComfyUI Primitive Mesh - Developer Guide

## Project Overview

**comfyui-primitive-mesh** is a ComfyUI custom node that converts images into artistic vector art using geometric shapes. It's a Python port of the [Sketchbeast](https://github.com/rossturk/sketchbeast) algorithm (derivative of [primitive.js](https://github.com/ondras/primitive.js)).

**Location:** `/home/rturk/comfy/ComfyUI/custom_nodes/comfyui-primitive-mesh`

## Architecture

### Core Files

- **`__init__.py`** - ComfyUI node registration
  - Registers `PrimitiveMeshNode` with ComfyUI
  - Sets up web directory for JavaScript extensions
  - Entry point for the custom node

- **`node.py`** (336 lines) - Main ComfyUI interface
  - `PrimitiveMeshNode` class - primary node implementation
  - Input parameters: image, num_shapes, shape_type, alpha, seed, compute_size, candidate_shapes, mutations
  - Outputs: IMAGE tensor and SVG string
  - Handles tensor conversions, scaling, progress callbacks, and preview generation
  - Category: `image/artistic`

- **`optimizer.py`** (185 lines) - Core optimization engine
  - `Optimizer` class - greedy shape optimization algorithm
  - Iteratively finds and applies best shapes using candidate generation + mutation refinement
  - Supports both sequential and parallel processing (sequential is default)
  - Progress callback system for ComfyUI integration

- **`shapes.py`** (484 lines) - Shape definitions
  - Abstract base class: `Shape`
  - Point-based base: `PointShape`
  - Concrete shapes: `Triangle`, `Rectangle`, `RotatedRectangle`, `Ellipse`, `Quadrilateral`
  - Each shape implements: `mutate()`, `render()`, `to_svg()`, `rasterize()`
  - SVG blur filter support

- **`state.py`** - Optimization state management
  - Tracks current canvas, target image, and distance metric

- **`step.py`** - Single optimization step
  - Represents one shape addition with color optimization
  - Computes optimal color and distance improvement

- **`util.py`** - Utility functions
  - Color computation and image difference calculations

### Test Files

- **`test_standalone.py`** - Standalone testing outside ComfyUI
- **`test_preview.py`** - Preview functionality testing

### Documentation

- **`README.md`** - Main documentation
- **`INSTALL.md`** - Installation instructions
- **`QUICK_START.md`** - Quick start guide
- **`COMFYUI_CONVERSION_SUMMARY.md`** - Notes on porting from JavaScript
- **`COMFYUI_PREVIEW_FIX.md`** - Preview system implementation details
- **`ALPHA_BLENDING_FIX.md`** - Alpha blending corrections

## Algorithm Flow

### High-Level Process

1. **Input Processing** (node.py:141-176)
   - Convert ComfyUI tensor (0-1 range) to numpy uint8 (0-255)
   - Scale down to `compute_size` for performance
   - Add alpha channel if missing
   - Compute fill color from border pixels (or use provided color)

2. **Optimization Loop** (optimizer.py:43-79)
   - For each shape iteration (1 to `num_shapes`):
     - Generate `candidate_shapes` random shapes
     - Compute optimal color for each
     - Select best candidate (lowest distance)
     - Refine through mutation (up to `mutations` attempts)
     - Apply if it improves overall state
     - Send progress callback with preview data

3. **Shape Optimization** (optimizer.py:157-184)
   - Mutation loop: try random modifications
   - Accept mutation if distance improves
   - Continue until `mutations` consecutive failures
   - Returns best mutated version

4. **Output Generation** (node.py:248-263)
   - Resize result back to original dimensions
   - Convert to ComfyUI tensor format (0-1 range)
   - Generate complete SVG document from shape list
   - Return (IMAGE tensor, SVG string)

## Key Parameters

### Performance Tuning

**Fast Preview:**
- num_shapes: 20-50
- compute_size: 256
- candidate_shapes: 100
- mutations: 30

**Balanced (Recommended):**
- num_shapes: 50-100
- compute_size: 400
- candidate_shapes: 200
- mutations: 50

**High Quality:**
- num_shapes: 150-300
- compute_size: 512-1024
- candidate_shapes: 300-500
- mutations: 100-200

### Shape Types

- **mixed** - All shapes (Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral)
- **rectangles** - RotatedRectangle only
- **triangles** - Triangle only
- **quadrilaterals** - Quadrilateral (diamond/rhombus) only
- **ellipses** - Ellipse only

## ComfyUI Integration

### Input/Output Types

**Inputs:**
- `image`: ComfyUI IMAGE tensor (B, H, W, C) in 0-1 range
- All other parameters are INT/FLOAT/STRING

**Outputs:**
- `image`: IMAGE tensor (1, H, W, C) in 0-1 range
- `svg`: STRING containing complete SVG document

### Progress & Preview System

The node integrates with ComfyUI's progress system:

1. **Progress Bar** (node.py:207)
   - Uses `comfy.utils.ProgressBar` if available
   - Updates on each shape iteration

2. **WebSocket Previews** (node.py:221-239)
   - Sends intermediate results via `PromptServer.instance.send_sync()`
   - Event: `"primitivemesh.preview"`
   - Payload: base64-encoded PNG + step/total
   - Preview interval: ~20 updates total

3. **Console Output** (node.py:242)
   - Prints progress every 10 shapes
   - Final summary with shape count

## Distance Metric

The optimization uses RMS (root-mean-square) difference in RGB space:
- Computed over all pixels in the difference image
- Normalized by image dimensions
- Lower distance = better approximation

## Shape Mutation Strategies

### PointShape (Triangle, Rectangle, Quadrilateral)
- Randomly selects one point
- Moves it by random angle/radius (max 10 pixels)

### RotatedRectangle
- Randomly chooses: rotation, x-scale, or y-scale
- Rotation: ±0.30 radians
- Scale: 0.8-1.2x multiplier

### Ellipse
- Randomly chooses: move center, scale rx, or scale ry
- Center: random angle/radius (max 20 pixels)
- Radii: ±20 pixels

## Development Patterns

### Adding New Shape Types

1. Create new class inheriting from `Shape` or `PointShape`
2. Implement required methods:
   - `__init__()` - initialize with random parameters
   - `mutate()` - return mutated copy
   - `render()` - draw to PIL ImageDraw
   - `_render_translated()` - draw with offset for rasterization
   - `to_svg()` - generate SVG markup
   - `compute_bbox()` - calculate bounding box
3. Add to `ALL_SHAPE_TYPES` in shapes.py
4. Update node.py INPUT_TYPES shape_type choices

### Testing Changes

**Standalone mode:**
```bash
python test_standalone.py
```

**With ComfyUI:**
1. Restart ComfyUI server
2. Add node to workflow
3. Check console for errors

### Common Gotchas

1. **Seed clamping** (node.py:137): NumPy requires seed ≤ 2^32-1, so we clamp with modulo
2. **Alpha channel** (node.py:159-162): Always ensure RGBA format for processing
3. **Tensor format** (node.py:142, 253): ComfyUI uses 0-1 range, algorithms use 0-255
4. **Bounding box** (shapes.py:107-125): Must be at least 1x1 to avoid PIL errors
5. **Progress callbacks** (node.py:209-243): Must handle None step when no improvement

## Dependencies

```
numpy>=1.21.0
Pillow>=9.0.0
torch>=1.13.0
```

ComfyUI-specific imports are optional (graceful degradation for standalone mode).

## GPU Acceleration

**Status**: ✅ Implemented (experimental)

The project now includes CUDA/GPU acceleration support:

- **New Files**: `util_torch.py`, `optimizer_torch.py`, `state_torch.py`, `step_torch.py`, `rasterizer_torch.py`
- **Performance**: ~1-5% speedup with current hybrid CPU/GPU approach
- **Device Selection**: Automatic (uses CUDA if available, falls back to CPU)
- **Node Parameter**: `use_gpu` toggle (default: True if CUDA available)

### Why Limited Speedup?

The bottleneck is **shape rasterization** (PIL/CPU), not color computation:
- Shape rasterization: 60-70% of time (CPU-bound, not accelerated)
- Color computation: 20-30% of time (GPU-accelerated ✅)
- Other operations: 5-10% (GPU-accelerated ✅)

### Potential Future Optimizations

For 10-20x speedup would need:
1. GPU-native shape rasterization (nvdiffrast or custom CUDA kernels)
2. Batch parallel candidate evaluation
3. Mixed precision (FP16)
4. Memory pre-allocation

See `GPU_ACCELERATION.md` for detailed analysis.

### Testing

```bash
source ~/.local/share/pipx/venvs/comfy-cli/bin/activate
python test_gpu.py
```

## Git Repository

- **Branch:** main
- **Recent commits:**
  - ca0e5d0 rename
  - f436358 docs
  - 3932a70 updates
  - 1134d50 first commit

## Future Improvements

From README.md roadmap:
- [ ] Line-based shapes (Squiggle, Scribble, Line, BentLine)
- [ ] Blur effects (SVG filters are defined but not heavily used)
- [ ] Multiprocessing for parallel shape evaluation
- [ ] Preview/intermediate output during processing (basic version implemented)
- [ ] Performance optimization with Cython
- [ ] More shape types (circles, stars, custom polygons)
- [ ] Animation/morphing support

## Debugging Tips

### Node Not Appearing
- Check ComfyUI console for import errors
- Verify `__init__.py` registration
- Restart ComfyUI completely

### Poor Quality Results
- Increase num_shapes
- Increase candidate_shapes and mutations
- Use higher compute_size
- Adjust alpha (0.4-0.6 often works well)

### Performance Issues
- Reduce compute_size to 256 or 128
- Lower candidate_shapes to 100
- Reduce mutations to 30
- Use fewer shapes

### Memory Issues
- Reduce compute_size
- Process smaller batches
- Monitor with system tools during optimization

## License

MIT License - matching original Sketchbeast project
- Original: Ondřej Žára (primitive.js)
- Derivative: Ross Turk (Sketchbeast)
- ComfyUI port: 2025
