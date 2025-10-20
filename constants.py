"""
Constants and configuration values for the primitive mesh optimizer.
"""

# Optimization settings
DEFAULT_COMPUTE_SIZE = 256  # Maximum dimension for optimization (smaller = faster)
PREVIEW_INTERVAL_DIVISOR = 20  # Show approximately this many previews total
PRINT_PROGRESS_INTERVAL = 10  # Print progress message every N shapes

# Alpha (transparency) constraints
MIN_ALPHA = 0.1
MAX_ALPHA = 1.0
ALPHA_MUTATION_RANGE = 0.08  # Maximum alpha change during mutation

# Style presets
STYLE_CRISPY_ALPHA_BASE = 0.85
STYLE_CRISPY_ALPHA_RANGE = 0.15  # 0.7 to 1.0

STYLE_DREAMY_ALPHA_BASE = 0.5
STYLE_DREAMY_ALPHA_RANGE = 0.3  # 0.35 to 0.65

STYLE_BLURRY_ALPHA_BASE = 0.25
STYLE_BLURRY_ALPHA_RANGE = 0.2  # 0.15 to 0.35

# Shape generation settings
SHAPE_MUTATION_DISTANCE = 10  # Pixel radius for point mutations
SHAPE_MUTATION_ANGLE_RANGE = 0.60  # Max angle change for rotated rectangles (radians)
SHAPE_MUTATION_SCALE_MIN = 0.8  # Min scale factor for size mutations
SHAPE_MUTATION_SCALE_MAX = 1.2  # Max scale factor (0.4 range, so 0.8 to 1.2)

# Triangle generation
TRIANGLE_INITIAL_RADIUS = 20  # Pixels from first point

# Rectangle generation
RECTANGLE_MUTATION_AMOUNT = 20  # Pixels to shift sides

# Rotated rectangle generation
ROTATED_RECT_MIN_SIZE = 1
ROTATED_RECT_MAX_SIZE = 300

# Ellipse generation
ELLIPSE_MIN_RADIUS = 1
ELLIPSE_MAX_RADIUS = 20
ELLIPSE_MUTATION_RADIUS = 20  # Pixels to move center
ELLIPSE_RADIUS_MUTATION = 20  # Max radius change

# Quadrilateral generation
QUAD_MIN_LENGTH = 1
QUAD_MAX_LENGTH = 75
QUAD_LENGTH_VARIATION = 0.4  # Second length as fraction of first
QUAD_LENGTH_OFFSET = 0.8  # Offset for second length calculation

# Color computation
COLOR_MIN = 0
COLOR_MAX = 255
RGB_CHANNELS = 3
