# ComfyUI Sketchbeast - Installation Guide

## Prerequisites

- ComfyUI installed and working
- Python 3.8 or higher
- pip package manager

## Installation Steps

### Step 1: Locate Your ComfyUI Installation

Find your ComfyUI installation directory. Common locations:
- **Windows**: `C:\ComfyUI` or `C:\Users\YourName\ComfyUI`
- **macOS**: `~/ComfyUI` or `~/Applications/ComfyUI`
- **Linux**: `~/ComfyUI` or `/opt/ComfyUI`

The custom nodes directory is: `ComfyUI/custom_nodes/`

### Step 2: Copy Files

Copy the `comfyui_primitivemesh` directory to your ComfyUI custom nodes folder:

```bash
# From the primitivemesh repository root
cp -r comfyui_primitivemesh /path/to/ComfyUI/custom_nodes/
```

Or use a symbolic link for development:

```bash
ln -s /path/to/sketchbeast/comfyui_primitivemesh /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh
```

### Step 3: Install Dependencies

Navigate to the node directory and install Python dependencies:

```bash
cd /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh
pip install -r requirements.txt
```

**Or** if using a virtual environment (recommended):

```bash
# Activate ComfyUI's virtual environment first
source /path/to/ComfyUI/venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Then install
cd custom_nodes/comfyui_primitivemesh
pip install -r requirements.txt
```

### Step 4: Restart ComfyUI

1. Stop ComfyUI if it's running
2. Start ComfyUI again
3. Check the console for any error messages

### Step 5: Verify Installation

1. Open ComfyUI in your browser
2. Right-click on the canvas to add a node
3. Search for "Sketchbeast" or navigate to: `Add Node > image > artistic > Sketchbeast Image to Vector Art`
4. If the node appears, installation was successful!

## Troubleshooting

### Node doesn't appear in ComfyUI

**Check console output:**
Look for error messages in the ComfyUI console when it starts. Common issues:

1. **Import errors**: Missing dependencies
   ```
   Solution: pip install -r requirements.txt
   ```

2. **Syntax errors**: File corruption or incomplete copy
   ```
   Solution: Re-copy the files or check file integrity
   ```

3. **Python version**: Too old
   ```
   Solution: Use Python 3.8 or newer
   ```

**Verify file structure:**
Your directory should look like this:
```
ComfyUI/
└── custom_nodes/
    └── comfyui_primitivemesh/
        ├── __init__.py
        ├── node.py
        ├── optimizer.py
        ├── shapes.py
        ├── state.py
        ├── step.py
        ├── util.py
        ├── requirements.txt
        ├── README.md
        ├── INSTALL.md
        └── test_standalone.py
```

### Dependency conflicts

If you get version conflicts with existing packages:

1. Try installing with `--upgrade`:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. Or create an isolated environment (advanced)

### Import errors at runtime

If the node loads but crashes when used:

1. Check that NumPy, Pillow, and PyTorch are compatible versions
2. Update to latest stable versions:
   ```bash
   pip install --upgrade numpy pillow torch
   ```

### Performance issues

If generation is very slow:

1. Reduce `compute_size` to 256 or 128
2. Reduce `num_shapes` to 20-50
3. Reduce `candidate_shapes` to 100
4. Check CPU usage - the algorithm is CPU-intensive

## Testing Without ComfyUI

To test the algorithm standalone (useful for debugging):

```bash
cd /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh

# Make sure dependencies are installed
pip install numpy pillow

# Run test (creates test image with colored circles)
python3 test_standalone.py --shapes 20 --output test_output.png --svg test_output.svg

# Or with your own image
python3 test_standalone.py --input your_image.jpg --shapes 50 --output result.png
```

## Updating

To update to a newer version:

1. Stop ComfyUI
2. Replace the `comfyui_primitivemesh` directory with the new version
3. Update dependencies: `pip install -r requirements.txt --upgrade`
4. Restart ComfyUI

## Uninstalling

To remove the node:

1. Stop ComfyUI
2. Delete the `comfyui_primitivemesh` directory:
   ```bash
   rm -rf /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh
   ```
3. Restart ComfyUI

## Getting Help

If you encounter issues:

1. Check the [README.md](README.md) for parameter guidance
2. Review the console output for error messages
3. Try the standalone test script to isolate the issue
4. Report issues at: https://github.com/rossturk/comfyui_primitivemesh/issues

## Advanced: Development Setup

For developers who want to modify the code:

1. Clone the repository:
   ```bash
   git clone https://github.com/rossturk/comfyui_primitivemesh.git
   cd comfyui_primitivemesh
   ```

2. Create symlink to ComfyUI:
   ```bash
   ln -s $(pwd)/comfyui_primitivemesh /path/to/ComfyUI/custom_nodes/comfyui_primitivemesh
   ```

3. Make changes to the code

4. Test with standalone script:
   ```bash
   cd comfyui_primitivemesh
   python3 test_standalone.py --shapes 10
   ```

5. Restart ComfyUI to see changes

Hot reload is not supported - you must restart ComfyUI after code changes.
