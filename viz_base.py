"""Renderer interface, shared branding and colour themes.

Renderers see only arrays (coordinates, connectivity, values), so they are
independent of geometry and of the FE code.
"""
import os
from dataclasses import dataclass

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Logos shown in the top-right corner, left to right, all with the same height
# (Scene.logo_height). Each entry maps a theme name to an image file in ASSETS
# ("light" is the fallback for other themes); an optional "scale" multiplies
# the common height for that logo. Missing files are skipped.
LOGOS = [
    {"light": "fsv_logo.png", "dark": "fsv_logo_dark.png"},  # Faculty of Civil Engineering, CTU
    # OOFEM: filled wordmark, readable on light and dark; very wide (8.6:1), so
    # it is drawn at 60 % of the common height to balance the faculty logo
    {"light": "oofem_logo.png", "scale": 0.6},
]

# Keys that close the visualization window and end the main loop
QUIT_KEYS = ("q", "Escape")

BRAND_TEXT = "(C) Katedra mechaniky, Fakulta stavební ČVUT"
WINDOW_TITLE = "OOFEM real-time demo · oofem.org"


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    foreground: str       # text, axes, colour bar labels
    edges: str            # element edges
    outline: str          # undeformed outline
    support: str          # support markers
    load: str             # load point marker
    cmap_signed: str      # diverging colour map for signed fields
    cmap_positive: str    # sequential colour map for non-negative fields


THEMES = {
    "light": Theme("light", background="#ffffff", foreground="#1a1a1a", edges="#404040",
                   outline="#9a9a9a", support="#1f4e9c", load="#c0392b",
                   cmap_signed="RdBu_r", cmap_positive="viridis"),
    "dark": Theme("dark", background="#16181d", foreground="#e8e8e8", edges="#202020",
                  outline="#6b6b6b", support="#6fa8ff", load="#ff6b5b",
                  cmap_signed="RdBu_r", cmap_positive="inferno"),
}


def logo_images(theme):
    """Return [(path, aspect=width/height, scale)] of available logos for the theme."""
    import matplotlib.image as mpimg
    out = []
    for entry in LOGOS:
        for name in (entry.get(theme.name), entry.get("light")):
            path = os.path.join(ASSETS, name) if name else None
            if path and os.path.isfile(path):
                h, w = mpimg.imread(path).shape[:2]
                out.append((path, w / h, entry.get("scale", 1.0)))
                break
    return out


@dataclass
class Scene:
    """Everything a renderer needs to set up the (static part of the) view."""
    coords: object        # (nn, 2) undeformed coordinates
    cells: object         # (ne, nen) 0-based connectivity
    cell_type: str        # "quad" | "tri"
    field_label: str      # e.g. "σxx [MPa]"
    clim: tuple           # fixed colour range
    signed: bool          # use diverging colour map
    title: str
    groups: dict          # named node groups, e.g. supports / load
    warp_margin: float    # extra view margin for the deformed shape [length]
    theme: Theme
    smooth: bool = False  # nodal (contour-like) instead of constant per-cell colouring
    show_mesh: bool = True   # draw element edges (the FE grid)
    mesh_alpha: float = 0.35  # opacity of the element edges (0 = invisible, 1 = opaque)
    logo_height: int = 46     # common logo height [px]; the faculty logo text needs >= ~44 px
    slider: object = None  # optional (label, vmin, vmax, callback) for manual input


class Renderer:
    def setup(self, scene):
        raise NotImplementedError

    def update(self, def_coords, cell_values, node_values, status):
        """Redraw with deformed coordinates and new values; status is a text line."""
        raise NotImplementedError

    def idle(self):
        """Keep the window responsive (events, resize) without new data."""

    def is_open(self):
        raise NotImplementedError

    def close(self):
        pass


def make_renderer(name):
    if name == "pv":
        from viz_pv import PyVistaRenderer
        return PyVistaRenderer()
    if name == "mpl":
        from viz_mpl import MatplotlibRenderer
        return MatplotlibRenderer()
    raise ValueError(f"unknown backend {name}")
