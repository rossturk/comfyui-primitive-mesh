"""
Shape classes for generating and mutating geometric primitives.
Ported from shape.js to Python.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw
import random
import math


class Shape(ABC):
    """Base class for all shapes."""

    def __init__(self, cfg: dict, w: int, h: int):
        self.cfg = cfg
        self.bbox = {}
        self.type = "Shape"
        self.color = (255, 255, 255)  # RGB tuple
        self.alpha = cfg.get('alpha', 0.5)

    @staticmethod
    def random_point(width: int, height: int) -> Tuple[int, int]:
        """Generate random point within bounds."""
        return (int(random.random() * width), int(random.random() * height))

    @staticmethod
    def create(cfg: dict) -> 'Shape':
        """Create random shape based on configuration."""
        shape_types = cfg['shapeTypes']
        shape_class = random.choice(shape_types)
        return shape_class(cfg, cfg['width'], cfg['height'])

    @abstractmethod
    def mutate(self, cfg: dict) -> 'Shape':
        """Return mutated version of this shape."""
        return self

    @abstractmethod
    def render(self, draw: ImageDraw.ImageDraw):
        """Render shape to ImageDraw context."""
        pass

    @abstractmethod
    def to_svg(self) -> str:
        """Generate SVG representation of shape."""
        pass

    @abstractmethod
    def scale(self, scale_factor: float) -> 'Shape':
        """Scale shape coordinates by a factor."""
        pass

    def rasterize(self, alpha: float) -> np.ndarray:
        """
        Rasterize shape to numpy array.

        Returns:
            RGBA image array with shape rendered in black
        """
        width = int(self.bbox.get('width', 1))
        height = int(self.bbox.get('height', 1))

        # Create image and draw context
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Translate by negative offset and render
        self._render_translated(draw, -self.bbox['left'], -self.bbox['top'], alpha)

        # Convert to numpy array
        return np.array(img, dtype=np.uint8)

    def _render_translated(self, draw: ImageDraw.ImageDraw, tx: int, ty: int, alpha: float):
        """Render with translation offset."""
        # This will be overridden by subclasses as needed
        pass

    def get_blur(self, blur: int) -> Optional[str]:
        """Get SVG blur filter reference."""
        if blur == 1:  # pleasing
            if random.randint(0, 1) == 1:
                choice = random.randint(0, 2)
                if choice == 0:
                    return "url(#g0.6)"
                elif choice == 1:
                    return ""
                else:
                    return "url(#g1)"
            return None
        elif blur == 2:  # dreamy
            choice = random.randint(0, 2)
            if choice == 0:
                return ""
            elif choice == 1:
                return "url(#g1)"
            else:
                return "url(#g10)"
        return None


class PointShape(Shape):
    """Base class for shapes defined by points."""

    def __init__(self, cfg: dict, w: int, h: int):
        super().__init__(cfg, w, h)
        self.points: List[Tuple[int, int]] = []
        self.linewidth = random.random() * (cfg.get('maxlinewidth', 2) - cfg.get('minlinewidth', 1)) + cfg.get('minlinewidth', 1)

    def compute_bbox(self):
        """Compute bounding box from points."""
        if not self.points:
            self.bbox = {'left': 0, 'top': 0, 'width': 1, 'height': 1}
            return self

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        self.bbox = {
            'left': int(min_x),
            'top': int(min_y),
            'width': max(int(max_x - min_x), 1),
            'height': max(int(max_y - min_y), 1)
        }
        return self

    def mutate(self, cfg: dict) -> 'PointShape':
        """Mutate by moving a random point."""
        clone = self.__class__(cfg, 0, 0)
        clone.points = [p[:] if isinstance(p, list) else (p[0], p[1]) for p in self.points]

        index = random.randint(0, len(self.points) - 1)
        point = list(clone.points[index])

        angle = random.random() * 2 * math.pi
        radius = random.random() * 10

        point[0] += int(radius * math.cos(angle))
        point[1] += int(radius * math.sin(angle))

        clone.points[index] = tuple(point)
        return clone.compute_bbox()


class Triangle(PointShape):
    """Triangle shape."""

    def __init__(self, cfg: dict, w: int, h: int):
        super().__init__(cfg, w, h)
        self.type = "Triangle"
        self.points = self._create_points(w, h, 3)
        self.compute_bbox()

    def _create_points(self, w: int, h: int, count: int) -> List[Tuple[int, int]]:
        """Create triangle points."""
        first = Shape.random_point(w, h)
        points = [first]

        for i in range(1, count):
            angle = random.random() * 2 * math.pi
            radius = random.random() * 20
            points.append((
                first[0] + int(radius * math.cos(angle)),
                first[1] + int(radius * math.sin(angle))
            ))

        return points

    def render(self, draw: ImageDraw.ImageDraw):
        """Render triangle."""
        if len(self.points) >= 3:
            draw.polygon(self.points, fill=self.color)

    def _render_translated(self, draw: ImageDraw.ImageDraw, tx: int, ty: int, alpha: float):
        """Render with translation."""
        translated = [(p[0] + tx, p[1] + ty) for p in self.points]
        alpha_int = int(alpha * 255)
        draw.polygon(translated, fill=(0, 0, 0, alpha_int))

    def to_svg(self) -> str:
        """Generate SVG path for triangle."""
        d = "M{},{}".format(self.points[0][0], self.points[0][1])
        for p in self.points[1:]:
            d += " L{},{}".format(p[0], p[1])
        d += "Z"

        blur = self.get_blur(self.cfg.get('blur', 0))
        filter_attr = f' filter="{blur}"' if blur else ''

        return f'<path d="{d}" fill="rgb({self.color[0]},{self.color[1]},{self.color[2]})" fill-opacity="{self.alpha:.2f}"{filter_attr}/>'

    def scale(self, scale_factor: float) -> 'Triangle':
        """Scale triangle coordinates."""
        self.points = [(int(p[0] * scale_factor), int(p[1] * scale_factor)) for p in self.points]
        self.compute_bbox()
        return self


class Rectangle(PointShape):
    """Axis-aligned rectangle shape."""

    def __init__(self, cfg: dict, w: int, h: int):
        super().__init__(cfg, w, h)
        self.type = "Rectangle"
        self.points = self._create_points(w, h)
        self.compute_bbox()

    def _create_points(self, w: int, h: int) -> List[Tuple[int, int]]:
        """Create rectangle points."""
        p1 = Shape.random_point(w, h)
        p2 = Shape.random_point(w, h)

        left = min(p1[0], p2[0])
        right = max(p1[0], p2[0])
        top = min(p1[1], p2[1])
        bottom = max(p1[1], p2[1])

        return [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom)
        ]

    def mutate(self, cfg: dict) -> 'Rectangle':
        """Mutate by adjusting one side."""
        clone = Rectangle(cfg, 0, 0)
        clone.points = [p for p in self.points]

        amount = int((random.random() - 0.5) * 20)
        side = random.randint(0, 3)

        points = [list(p) for p in clone.points]

        if side == 0:  # left
            points[0][0] += amount
            points[3][0] += amount
        elif side == 1:  # top
            points[0][1] += amount
            points[1][1] += amount
        elif side == 2:  # right
            points[1][0] += amount
            points[2][0] += amount
        else:  # bottom
            points[2][1] += amount
            points[3][1] += amount

        clone.points = [tuple(p) for p in points]
        return clone.compute_bbox()

    def render(self, draw: ImageDraw.ImageDraw):
        """Render rectangle."""
        if len(self.points) >= 4:
            draw.polygon(self.points, fill=self.color)

    def _render_translated(self, draw: ImageDraw.ImageDraw, tx: int, ty: int, alpha: float):
        """Render with translation."""
        translated = [(p[0] + tx, p[1] + ty) for p in self.points]
        alpha_int = int(alpha * 255)
        draw.polygon(translated, fill=(0, 0, 0, alpha_int))

    def to_svg(self) -> str:
        """Generate SVG path for rectangle."""
        d = "M{},{}".format(self.points[0][0], self.points[0][1])
        for p in self.points[1:]:
            d += " L{},{}".format(p[0], p[1])
        d += "Z"

        blur = self.get_blur(self.cfg.get('blur', 0))
        filter_attr = f' filter="{blur}"' if blur else ''

        return f'<path d="{d}" fill="rgb({self.color[0]},{self.color[1]},{self.color[2]})" fill-opacity="{self.alpha:.2f}"{filter_attr}/>'

    def scale(self, scale_factor: float) -> 'Rectangle':
        """Scale rectangle coordinates."""
        self.points = [(int(p[0] * scale_factor), int(p[1] * scale_factor)) for p in self.points]
        self.compute_bbox()
        return self


class RotatedRectangle(PointShape):
    """Rotated rectangle shape."""

    def __init__(self, cfg: dict, w: int, h: int):
        super().__init__(cfg, w, h)
        self.type = "RotatedRectangle"
        self.center = Shape.random_point(w, h)
        self.sizex = 1 + random.random() * 300
        self.sizey = 1 + random.random() * 300
        self.angle = random.random() * math.pi
        self.points = self._create_rect_points(self.center, self.sizex, self.sizey, self.angle)
        self.compute_bbox()

    def _create_rect_points(self, center: Tuple[int, int], w: float, h: float, angle: float) -> List[Tuple[int, int]]:
        """Create rotated rectangle points."""
        sin_ang = math.sin(angle)
        cos_ang = math.cos(angle)

        up_diff = sin_ang * w
        side_diff = cos_ang * w

        points = [center]
        points.append((int(center[0] + side_diff), int(center[1] + up_diff)))

        up_diff = cos_ang * h
        side_diff = sin_ang * h

        points.append((int(points[1][0] + side_diff), int(points[1][1] - up_diff)))
        points.append((int(center[0] + side_diff), int(center[1] - up_diff)))

        return points

    def mutate(self, cfg: dict) -> 'RotatedRectangle':
        """Mutate rotation or size."""
        clone = RotatedRectangle(cfg, 0, 0)
        clone.angle = self.angle
        clone.sizex = self.sizex
        clone.sizey = self.sizey
        clone.center = self.center

        choice = random.randint(0, 2)
        if choice == 0:  # rotate
            clone.angle += random.random() * 0.60 - 0.30
        elif choice == 1:  # scale x
            clone.sizex *= random.random() * 0.4 + 0.8
        else:  # scale y
            clone.sizey *= random.random() * 0.4 + 0.8

        clone.points = self._create_rect_points(clone.center, clone.sizex, clone.sizey, clone.angle)
        return clone.compute_bbox()

    def render(self, draw: ImageDraw.ImageDraw):
        """Render rotated rectangle."""
        if len(self.points) >= 4:
            draw.polygon(self.points, fill=self.color)

    def _render_translated(self, draw: ImageDraw.ImageDraw, tx: int, ty: int, alpha: float):
        """Render with translation."""
        translated = [(p[0] + tx, p[1] + ty) for p in self.points]
        alpha_int = int(alpha * 255)
        draw.polygon(translated, fill=(0, 0, 0, alpha_int))

    def to_svg(self) -> str:
        """Generate SVG path for rotated rectangle."""
        d = "M{},{}".format(self.points[0][0], self.points[0][1])
        for p in self.points[1:]:
            d += " L{},{}".format(p[0], p[1])
        d += "Z"

        blur = self.get_blur(self.cfg.get('blur', 0))
        filter_attr = f' filter="{blur}"' if blur else ''

        return f'<path d="{d}" fill="rgb({self.color[0]},{self.color[1]},{self.color[2]})" fill-opacity="{self.alpha:.2f}"{filter_attr}/>'

    def scale(self, scale_factor: float) -> 'RotatedRectangle':
        """Scale rotated rectangle coordinates."""
        self.center = (int(self.center[0] * scale_factor), int(self.center[1] * scale_factor))
        self.sizex *= scale_factor
        self.sizey *= scale_factor
        self.points = self._create_rect_points(self.center, self.sizex, self.sizey, self.angle)
        self.compute_bbox()
        return self


class Ellipse(Shape):
    """Ellipse shape."""

    def __init__(self, cfg: dict, w: int, h: int):
        super().__init__(cfg, w, h)
        self.type = "Ellipse"
        self.center = Shape.random_point(w, h)
        self.rx = 1 + int(random.random() * 20)
        self.ry = 1 + int(random.random() * 20)
        self.rot = random.random() * math.pi / 2
        self.compute_bbox()

    def compute_bbox(self):
        """Compute bounding box for ellipse."""
        rmax = max(self.rx, self.ry)
        self.bbox = {
            'left': self.center[0] - rmax,
            'top': self.center[1] - rmax,
            'width': 2 * rmax,
            'height': 2 * rmax
        }
        return self

    def mutate(self, cfg: dict) -> 'Ellipse':
        """Mutate ellipse parameters."""
        clone = Ellipse(cfg, 0, 0)
        clone.center = self.center
        clone.rx = self.rx
        clone.ry = self.ry
        clone.rot = self.rot

        choice = random.randint(0, 2)
        if choice == 0:  # move center
            angle = random.random() * 2 * math.pi
            radius = random.random() * 20
            clone.center = (
                int(clone.center[0] + radius * math.cos(angle)),
                int(clone.center[1] + radius * math.sin(angle))
            )
        elif choice == 1:  # scale rx
            clone.rx += int((random.random() - 0.5) * 20)
            clone.rx = max(1, clone.rx)
        else:  # scale ry
            clone.ry += int((random.random() - 0.5) * 20)
            clone.ry = max(1, clone.ry)

        return clone.compute_bbox()

    def render(self, draw: ImageDraw.ImageDraw):
        """Render ellipse with rotation support."""
        # If no rotation, use simple ellipse
        if abs(self.rot) < 0.01:
            bbox = [
                self.center[0] - self.rx,
                self.center[1] - self.ry,
                self.center[0] + self.rx,
                self.center[1] + self.ry
            ]
            draw.ellipse(bbox, fill=self.color)
        else:
            # For rotated ellipses, approximate with a polygon
            points = self._get_rotated_ellipse_points()
            draw.polygon(points, fill=self.color)

    def _get_rotated_ellipse_points(self, num_points: int = 64):
        """Generate points for a rotated ellipse."""
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # Point on unrotated ellipse
            x = self.rx * math.cos(angle)
            y = self.ry * math.sin(angle)
            # Rotate by self.rot
            x_rot = x * math.cos(self.rot) - y * math.sin(self.rot)
            y_rot = x * math.sin(self.rot) + y * math.cos(self.rot)
            # Translate to center
            points.append((
                int(self.center[0] + x_rot),
                int(self.center[1] + y_rot)
            ))
        return points

    def _render_translated(self, draw: ImageDraw.ImageDraw, tx: int, ty: int, alpha: float):
        """Render with translation."""
        alpha_int = int(alpha * 255)

        # If no rotation, use simple ellipse
        if abs(self.rot) < 0.01:
            bbox = [
                self.center[0] - self.rx + tx,
                self.center[1] - self.ry + ty,
                self.center[0] + self.rx + tx,
                self.center[1] + self.ry + ty
            ]
            draw.ellipse(bbox, fill=(0, 0, 0, alpha_int))
        else:
            # For rotated ellipses, use polygon
            points = self._get_rotated_ellipse_points()
            translated = [(p[0] + tx, p[1] + ty) for p in points]
            draw.polygon(translated, fill=(0, 0, 0, alpha_int))

    def to_svg(self) -> str:
        """Generate SVG ellipse."""
        rot_deg = self.rot * (180 / math.pi)
        blur = self.get_blur(self.cfg.get('blur', 0))
        filter_attr = f' filter="{blur}"' if blur else ''

        return f'<ellipse cx="{self.center[0]}" cy="{self.center[1]}" rx="{self.rx}" ry="{self.ry}" fill="rgb({self.color[0]},{self.color[1]},{self.color[2]})" fill-opacity="{self.alpha:.2f}" transform="rotate({rot_deg:.2f},{self.center[0]},{self.center[1]})"{filter_attr}/>'

    def scale(self, scale_factor: float) -> 'Ellipse':
        """Scale ellipse coordinates."""
        self.center = (int(self.center[0] * scale_factor), int(self.center[1] * scale_factor))
        self.rx = int(self.rx * scale_factor)
        self.ry = int(self.ry * scale_factor)
        self.compute_bbox()
        return self


class Quadrilateral(PointShape):
    """Quadrilateral (rhombus) shape."""

    def __init__(self, cfg: dict, w: int, h: int):
        super().__init__(cfg, w, h)
        self.type = "Quadrilateral"
        self.points = self._create_points(w, h)
        self.compute_bbox()

    def _create_points(self, w: int, h: int) -> List[Tuple[int, int]]:
        """Create quadrilateral points."""
        center = Shape.random_point(w, h)
        l1 = 1 + random.random() * 75
        l2 = l1 + l1 * (random.random() * 0.4 - 0.8)
        angle = random.random() * math.pi

        points = []
        points.append((
            int(center[0] + l1 * math.cos(angle)),
            int(center[1] + l1 * math.sin(angle))
        ))
        points.append((
            int(center[0] + l2 * math.cos(angle + math.pi / 2)),
            int(center[1] + l2 * math.sin(angle + math.pi / 2))
        ))
        points.append((
            int(center[0] + l1 * math.cos(angle + math.pi)),
            int(center[1] + l1 * math.sin(angle + math.pi))
        ))
        points.append((
            int(center[0] + l2 * math.cos(angle + math.pi * 1.5)),
            int(center[1] + l2 * math.sin(angle + math.pi * 1.5))
        ))

        return points

    def render(self, draw: ImageDraw.ImageDraw):
        """Render quadrilateral."""
        if len(self.points) >= 4:
            draw.polygon(self.points, fill=self.color)

    def _render_translated(self, draw: ImageDraw.ImageDraw, tx: int, ty: int, alpha: float):
        """Render with translation."""
        translated = [(p[0] + tx, p[1] + ty) for p in self.points]
        alpha_int = int(alpha * 255)
        draw.polygon(translated, fill=(0, 0, 0, alpha_int))

    def to_svg(self) -> str:
        """Generate SVG path for quadrilateral."""
        d = "M{},{}".format(self.points[0][0], self.points[0][1])
        for p in self.points[1:]:
            d += " L{},{}".format(p[0], p[1])
        d += "Z"

        blur = self.get_blur(self.cfg.get('blur', 0))
        filter_attr = f' filter="{blur}"' if blur else ''

        return f'<path d="{d}" fill="rgb({self.color[0]},{self.color[1]},{self.color[2]})" fill-opacity="{self.alpha:.2f}"{filter_attr}/>'

    def scale(self, scale_factor: float) -> 'Quadrilateral':
        """Scale quadrilateral coordinates."""
        self.points = [(int(p[0] * scale_factor), int(p[1] * scale_factor)) for p in self.points]
        self.compute_bbox()
        return self


# Shape type list for configuration
ALL_SHAPE_TYPES = [Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral]
