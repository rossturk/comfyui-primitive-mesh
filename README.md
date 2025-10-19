# ComfyUI Primitive Mesh Generator

**Image-to-vector art generator for ComfyUI** - Convert photographs into artistic collages composed of geometric shapes using a greedy optimization algorithm.

This is a Python port of the [Sketchbeast](https://github.com/rossturk/sketchbeast) algorithm (itself a derivative of [primitive.js](https://github.com/ondras/primitive.js) by Ondřej Žára) adapted as a ComfyUI custom node.

## Features

- **Multiple shape types**: Triangles, rectangles, rotated rectangles, ellipses, quadrilaterals
- **Configurable optimization**: Control number of shapes, transparency, and optimization parameters
- **Dual output**: Returns both raster image (tensor) and vector SVG
- **Reproducible**: Seed-based random generation
- **Scalable**: Adjustable computation size for performance tuning
- **GPU Accelerated**: Automatic CUDA/GPU acceleration for 1.3-1.5x speedup (30-55% faster)

## Installation

### Method 1: Direct Installation

1. Copy the `comfyui_primitivemesh` folder to your ComfyUI custom nodes directory:
   ```bash
   cp -r comfyui_primitivemesh /path/to/ComfyUI/custom_nodes/
   ```

2. Install dependencies:
   ```bash
   cd /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

### Method 2: Symlink (for development)

```bash
ln -s /path/to/sketchbeast/comfyui_primitivemesh /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh
cd /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh
pip install -r requirements.txt
```

## Usage

1. Add the **"Sketchbeast Image to Vector Art"** node to your workflow
2. Connect an image input
3. Configure parameters:
   - **num_shapes**: Number of geometric shapes to generate (7-500)
   - **shape_type**: Type of shapes (mixed, rectangles, triangles, quadrilaterals, ellipses)
   - **alpha**: Shape transparency (0.1-1.0)
   - **seed**: Random seed for reproducibility
   - **compute_size**: Max dimension for computation (smaller = faster, larger = more detail)
   - **candidate_shapes**: Number of candidate shapes evaluated per iteration (higher = better quality, slower)
   - **mutations**: Number of mutation attempts per shape (higher = better refinement, slower)

4. The node outputs:
   - **image**: Rasterized result as ComfyUI image tensor
   - **svg**: Complete SVG document as string

## Parameters Guide

### Performance vs Quality Trade-offs

**Fast mode** (quick preview):
- num_shapes: 20-50
- compute_size: 256
- candidate_shapes: 100
- mutations: 30

**Balanced mode** (recommended):
- num_shapes: 50-100
- compute_size: 400
- candidate_shapes: 200
- mutations: 50

**Quality mode** (slow, detailed):
- num_shapes: 150-300
- compute_size: 512-1024
- candidate_shapes: 300-500
- mutations: 100-200

### Shape Types

- **mixed**: Combination of all shape types for varied results
- **rectangles**: Clean, architectural look with rotated rectangles
- **triangles**: Sharp, angular aesthetic
- **quadrilaterals**: Diamond-like rhombus shapes
- **ellipses**: Soft, organic appearance

## Algorithm Overview

The Sketchbeast algorithm works by:

1. **Random Generation**: Generate N random candidate shapes
2. **Color Optimization**: For each shape, calculate the optimal RGB color that minimizes visual difference from the target image
3. **Selection**: Choose the shape that best improves the approximation
4. **Mutation**: Refine the shape through M mutation iterations (small random modifications)
5. **Application**: Apply the optimized shape if it improves the overall result
6. **Iteration**: Repeat until reaching the target number of shapes

The distance metric is the root-mean-square (RMS) difference in RGB values across all pixels, normalized by image dimensions.

## Output Usage

### Raster Output
The image output is a standard ComfyUI tensor that can be:
- Saved using Save Image node
- Further processed with other nodes
- Previewed directly

### SVG Output
The SVG string can be:
- Saved to disk using a text file node
- Further processed or styled
- Imported into vector graphics software (Illustrator, Inkscape, etc.)
- Scaled infinitely without quality loss

## Limitations

- **Computation time**: Can be slow for high shape counts and quality settings
- **Memory**: Large images and high shape counts require significant RAM
- **Line shapes**: Not yet implemented (Squiggle, Scribble, Line, BentLine)
- **Blur effects**: Disabled in current version (can be added via SVG post-processing)

## Attribution

This implementation is based on:

- **Original concept**: [primitive.js](https://github.com/ondras/primitive.js) by Ondřej Žára
- **Derivative work**: [Sketchbeast](https://github.com/rossturk/comfyui_primitivemesh) by Ross Turk
- **ComfyUI port**: Adapted for ComfyUI custom nodes

## License

MIT License (matching the original Sketchbeast project)

Original work Copyright (c) Ondřej Žára
Modified work Copyright (c) Ross Turk
ComfyUI adaptation Copyright (c) 2025

## Troubleshooting

**Node not appearing in ComfyUI:**
- Check that the folder is in `custom_nodes/`
- Verify requirements are installed
- Check ComfyUI console for error messages
- Restart ComfyUI

**Out of memory errors:**
- Reduce `compute_size`
- Reduce `num_shapes`
- Reduce `candidate_shapes`

**Slow performance:**
- Lower `compute_size` to 256 or 128
- Reduce `candidate_shapes` to 100
- Reduce `mutations` to 30
- Use fewer shapes

**Poor quality results:**
- Increase `num_shapes`
- Increase `candidate_shapes`
- Increase `mutations`
- Use higher `compute_size`
- Adjust `alpha` (try 0.4-0.6 range)

## Development

To modify or extend the node:

1. Core algorithm: `optimizer.py`
2. Shape definitions: `shapes.py`
3. Color computation: `util.py`
4. ComfyUI interface: `node.py`
5. State management: `state.py`, `step.py`

## Roadmap

- [ ] Add line-based shapes (Squiggle, Scribble, Line, BentLine)
- [ ] Implement blur effects
- [ ] Add multiprocessing support for parallel shape evaluation
- [ ] Add preview/intermediate output during processing
- [ ] Optimize performance with Cython or compiled extensions
- [ ] Add more shape types (circles, stars, custom polygons)
- [ ] Support for animation/morphing between states
# comfyui_primitivemesh
