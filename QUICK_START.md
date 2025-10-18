# ComfyUI Sketchbeast - Quick Start Guide

## 5-Minute Setup

### 1. Install
```bash
# Copy to ComfyUI
cp -r comfyui_primitivemesh /path/to/ComfyUI/custom_nodes/

# Install dependencies
cd /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh
pip install -r requirements.txt
```

### 2. Restart ComfyUI

### 3. Add Node
- Right-click canvas
- Navigate: `Add Node > image > artistic > Sketchbeast Image to Vector Art`

### 4. Connect & Configure
- Connect an image input
- Set parameters (or use defaults)
- Queue prompt

## Recommended Settings

### For Quick Preview
- **num_shapes**: 20
- **compute_size**: 256
- **candidate_shapes**: 100
- **mutations**: 30

### For Good Quality
- **num_shapes**: 50
- **compute_size**: 400
- **candidate_shapes**: 200
- **mutations**: 50

### For Best Quality
- **num_shapes**: 150
- **compute_size**: 512
- **candidate_shapes**: 300
- **mutations**: 100

## Common Workflows

### Basic Usage
```
Load Image → Sketchbeast → Preview Image
                        └→ Save Image
```

### With SVG Output
```
Load Image → Sketchbeast → Preview Image
                        └→ (SVG string output - save to text file)
```

### Batch Processing
```
Load Images (batch) → Sketchbeast → Save Image
```

## Tips

1. **Start small**: Begin with 20-30 shapes to test
2. **Use seed**: Set seed for reproducible results
3. **Adjust alpha**: 0.4-0.6 usually works best
4. **Shape types**:
   - `mixed` - varied, organic look
   - `triangles` - angular, geometric
   - `rectangles` - architectural, structured
   - `ellipses` - soft, rounded

## Troubleshooting

**Slow?** → Reduce compute_size and num_shapes

**Low quality?** → Increase compute_size and candidate_shapes

**Node missing?** → Check console, reinstall dependencies

**Out of memory?** → Reduce compute_size to 256 or 128

## Full Documentation

- Installation: [INSTALL.md](INSTALL.md)
- Parameters: [README.md](README.md)
- Testing: Run `test_standalone.py`

## Support

Issues: https://github.com/rossturk/comfyui_primitivemesh/issues
