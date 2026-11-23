"""PyVista (VTK) renderer: one UnstructuredGrid updated in place.

Supports constant per-cell colouring and smooth nodal colouring (optionally
banded, giving a contour-plot look).
"""
import os

# Under WSL, Mesa defaults to the llvmpipe software rasterizer; the d3d12
# driver renders on the Windows GPU (about 2x faster here). Must be set before
# the OpenGL context is created; an explicit user setting is kept.
if os.path.isdir("/usr/lib/wsl/lib"):
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")

import numpy as np
import pyvista as pv
from matplotlib import font_manager

try:
    from vtkmodules.vtkRenderingCore import VTK_FONT_FILE
except ImportError:  # older VTK layouts
    VTK_FONT_FILE = 4

from viz_base import BRAND_TEXT, QUIT_KEYS, WINDOW_TITLE, Renderer, logo_images

CELL_TYPES = {"quad": pv.CellType.QUAD, "tri": pv.CellType.TRIANGLE}

# VTK's built-in fonts lack Greek letters (σ, ε, γ); use a TrueType font that
# has them. DejaVu Sans ships with matplotlib, so it is always available.
FONT_FILE = font_manager.findfont("DejaVu Sans")
# monospace variant for the status text, whose labels and numbers are aligned in columns
MONO_FONT_FILE = font_manager.findfont("DejaVu Sans Mono")

# Fraction of the window width usable by the model (scalar bar sits to the right)
MODEL_VIEW_X = (0.02, 0.86)


def _use_font(prop, font_file=FONT_FILE):
    prop.SetFontFamily(VTK_FONT_FILE)
    prop.SetFontFile(font_file)


class PyVistaRenderer(Renderer):
    def __init__(self, n_bands=None):
        self.n_bands = n_bands      # number of discrete colour bands (None = continuous)

    def setup(self, sc):
        th = sc.theme
        self.sc = sc
        pts = np.column_stack([sc.coords, np.zeros(len(sc.coords))])
        grid = pv.UnstructuredGrid({CELL_TYPES[sc.cell_type]: sc.cells}, pts)
        if sc.smooth:
            grid.point_data["val"] = np.zeros(grid.n_points)
        else:
            grid.cell_data["val"] = np.zeros(grid.n_cells)
        self.grid = grid

        pl = pv.Plotter(window_size=(1500, 560), title=WINDOW_TITLE)
        pl.set_background(th.background)
        pl.add_mesh(grid.outline(), color=th.outline, line_width=1)
        pl.add_mesh(grid, scalars="val", preference="point" if sc.smooth else "cell",
                    cmap=th.cmap_signed if sc.signed else th.cmap_positive,
                    clim=sc.clim, show_edges=sc.show_mesh, edge_color=th.edges,
                    edge_opacity=sc.mesh_alpha,
                    line_width=0.5, interpolate_before_map=True, n_colors=self.n_bands or 256,
                    lighting=False,          # flat 2D view: show true colour-map colours
                    scalar_bar_args=dict(title=sc.field_label, color=th.foreground,
                                         vertical=True, position_x=0.90, position_y=0.15,
                                         height=0.7, width=0.04, fmt="%.3g"),
                    show_scalar_bar=True)

        sup = sc.groups.get("supports")
        if sup is not None and len(sup):
            cones = pv.PolyData(pts[sup] - np.array([0.0, 2.5, 0.0])).glyph(
                geom=pv.Cone(direction=(0, 1, 0), height=4.0, radius=2.5, resolution=3),
                scale=False, orient=False)
            pl.add_mesh(cones, color=th.support)
        self.load_ids = sc.groups.get("load")
        if self.load_ids is not None:
            self.load_pts = pv.PolyData(pts[self.load_ids] + np.array([0.0, 2.5, 0.0]))
            self.load_geom = pv.Cone(direction=(0, -1, 0), height=4.0, radius=2.5, resolution=3)
            self.load_glyph = self.load_pts.glyph(geom=self.load_geom, scale=False, orient=False)
            pl.add_mesh(self.load_glyph, color=th.load)

        texts = [pl.add_text(sc.title, position="upper_left", font_size=12, color=th.foreground),
                 pl.add_text("", position="lower_left", font_size=10, color=th.foreground),
                 pl.add_text(BRAND_TEXT, position="lower_right", font_size=10, color=th.foreground)]
        self.status = texts[1]
        for t in texts:
            _use_font(t.GetTextProperty(), MONO_FONT_FILE if t is self.status else FONT_FILE)
        for bar in pl.scalar_bars.values():
            _use_font(bar.GetTitleTextProperty())
            _use_font(bar.GetLabelTextProperty())

        self.slider = None
        if sc.slider:
            label, vmin, vmax, cb_fn = sc.slider
            # "always": call back on every move while dragging, not only on release
            self.slider = pl.add_slider_widget(cb_fn, rng=[vmin, vmax], value=vmin, title=label,
                                               color=th.foreground, style="modern",
                                               interaction_event="always", fmt="%.2f",
                                               tube_width=0.025, slider_width=0.03)
            rep = self.slider.GetRepresentation()
            _use_font(rep.GetTitleProperty())
            _use_font(rep.GetLabelProperty())
            rep.GetTitleProperty().BoldOff()

        self.pl = pl
        self._quit = False
        for key in QUIT_KEYS:
            pl.add_key_event(key, self._request_quit)
        self._logos = [(pl.add_logo_widget(path, position=(0.0, 0.0), size=(0.1, 0.1), opacity=1.0),
                        aspect, scale)
                       for path, aspect, scale in logo_images(th)]
        for widget, _, _ in self._logos:
            # static decoration: no hover border, cannot be dragged or resized
            widget.GetRepresentation().SetShowBorderToOff()
            widget.ProcessEventsOff()
        self._size = None
        self._layout()
        pl.show(interactive_update=True, auto_close=False)
        self._layout()              # the window may open with a different size

    def _layout(self):
        """Place logos and fit the camera for the current window size.

        Viewport-normalized positions depend on the window aspect ratio, so this
        is redone whenever the window is resized (checked in update()).
        """
        size = tuple(self.pl.window_size)
        if size == self._size or min(size) <= 0:
            return False
        self._size = size
        logo_bottom = self._place_logos(size)
        self._place_scalar_bar(size, top_px=logo_bottom - 24)   # room for the bar title
        bottom_px = self._place_slider(size) if self.slider is not None else 0
        self._fit_camera(self.pl, self.sc, size, bottom_px=bottom_px)
        return True

    # Slider geometry in pixels. VTK sizes the slider's title and value label
    # relative to the bar length, so a fixed pixel length keeps the text size
    # fixed when the window is resized.
    SLIDER_LENGTH_PX = 520
    SLIDER_Y_PX = 100           # bar centre: above the three-line status text and the bar title

    def _place_slider(self, size):
        """Put the slider in a strip above the status text, centred under the model.

        The title is drawn below the bar and the current value above it, so the
        strip spans roughly SLIDER_Y_PX +- 30 px. Returns the y pixel coordinate
        of the top of the strip, which the camera fit keeps free of the model.
        """
        ww, wh = size
        length = min(self.SLIDER_LENGTH_PX, 0.4 * ww)
        xc = 0.5 * sum(MODEL_VIEW_X) * ww
        rep = self.slider.GetRepresentation()
        rep.GetPoint1Coordinate().SetValue((xc - 0.5 * length) / ww, self.SLIDER_Y_PX / wh)
        rep.GetPoint2Coordinate().SetValue((xc + 0.5 * length) / ww, self.SLIDER_Y_PX / wh)
        return self.SLIDER_Y_PX + 0.06 * length

    def _place_scalar_bar(self, size, top_px, bottom_px=70):
        """Fit the vertical colour bar between the logos and the status text."""
        ww, wh = size
        top_px = min(top_px, wh - 10)
        for bar in self.pl.scalar_bars.values():
            bar.SetPosition(0.90, bottom_px / wh)
            bar.SetHeight(max(top_px - bottom_px, 40) / wh)

    @staticmethod
    def _fit_camera(pl, sc, size, pad=0.06, bottom_px=0):
        """Parallel projection fitted to the model plus room for deformation/markers.

        reset_camera() fits the bounding sphere, which leaves a slender beam
        tiny; here the visible world box is fitted to the usable window area:
        MODEL_VIEW_X horizontally, and vertically from bottom_px (at least 10 %
        of the height, kept free for the status text and slider) to 90 %.
        """
        (xmin, ymin), (xmax, ymax) = sc.coords.min(axis=0), sc.coords.max(axis=0)
        mark = 0.25 * (ymax - ymin) + 5.0          # room for support/load markers
        ymin -= sc.warp_margin + mark
        ymax += mark
        w, h = (xmax - xmin) * (1 + 2 * pad), (ymax - ymin) * (1 + 2 * pad)
        ww, wh = size
        aspect = ww / wh
        fx0, fx1 = MODEL_VIEW_X
        fy0, fy1 = min(max(0.1, bottom_px / wh), 0.6), 0.9
        ps = max(0.5 * h / (fy1 - fy0), 0.5 * w / (aspect * (fx1 - fx0)))   # half view height
        # shift the view so the model is centred in the usable x and y ranges
        cx = 0.5 * (xmin + xmax) + (0.5 - 0.5 * (fx0 + fx1)) * 2 * ps * aspect
        cy = 0.5 * (ymin + ymax) + (0.5 - 0.5 * (fy0 + fy1)) * 2 * ps
        pl.camera_position = [(cx, cy, 1000.0), (cx, cy, 0.0), (0.0, 1.0, 0.0)]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = ps

    def _place_logos(self, size, height_px=None, right_px=12, top_px=8, gap_px=14, max_width=0.38):
        """Right-align logos in the top-right corner with fixed pixel sizes.

        Sizes are given in pixels and converted to the viewport-normalized
        coordinates VTK expects, so the logos keep their size, aspect ratio and
        corner position when the window is resized. In narrow windows they are
        shrunk to at most max_width of the window width, leaving room for the
        title. Logos of different heights (LOGOS "scale") are centred vertically
        in the row. Returns the y pixel coordinate of the bottom of the logo row.
        """
        ww, wh = size
        height_px = height_px or self.sc.logo_height
        total = sum(height_px * s * a for _, a, s in self._logos) + gap_px * (len(self._logos) - 1)
        f = min(1.0, max_width * ww / total) if total > 0 else 1.0
        row_h = max((height_px * s * f for _, _, s in self._logos), default=0.0)
        row_bottom = wh - top_px - row_h
        x = ww - right_px
        for widget, aspect, scale in reversed(self._logos):
            h = height_px * scale * f
            w = h * aspect
            x -= w
            rep = widget.GetRepresentation()
            rep.SetPosition(x / ww, (row_bottom + 0.5 * (row_h - h)) / wh)
            rep.SetPosition2(w / ww, h / wh)
            x -= gap_px * f
        return row_bottom

    def update(self, def_coords, cell_values, node_values, status):
        self.grid.points[:, :2] = def_coords
        if self.sc.smooth:
            self.grid.point_data["val"][:] = node_values
        else:
            self.grid.cell_data["val"][:] = cell_values
        self.grid.Modified()
        if self.load_ids is not None:
            self.load_pts.points[:, :2] = def_coords[self.load_ids] + np.array([0.0, 2.5])
            self.load_pts.Modified()
            self.load_glyph.copy_from(self.load_pts.glyph(geom=self.load_geom, scale=False, orient=False))
        self._set_status(status)
        self._layout()                  # no-op unless the window was resized
        # One explicit render, then handle window events. Plotter.update()
        # installs a repeating timer that causes extra renders per frame.
        self.pl.render()
        self.pl.iren.process_events()

    def idle(self):
        if self._layout():              # window was resized: redraw once
            self.pl.render()
        self.pl.iren.process_events()

    def _set_status(self, text):
        st = self.status
        if hasattr(st, "SetText") and hasattr(st, "GetText") and st.GetClassName() == "vtkCornerAnnotation":
            st.SetText(0, text)          # 0 = lower left corner
        else:
            st.SetInput(text)

    def _request_quit(self):
        self._quit = True       # the main loop sees is_open() == False and closes

    def is_open(self):
        pl = getattr(self, "pl", None)
        return (pl is not None and not self._quit and not getattr(pl, "_closed", False)
                and pl.render_window is not None)

    def close(self):
        pl = getattr(self, "pl", None)
        if pl is not None and not getattr(pl, "_closed", False):
            pl.close()
