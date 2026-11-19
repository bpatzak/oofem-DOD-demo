"""PyVista (VTK) renderer: one UnstructuredGrid updated in place.

Supports constant per-cell colouring and smooth nodal colouring (optionally
banded, giving a contour-plot look).
"""
import numpy as np
import pyvista as pv

from viz_base import BRAND_TEXT, WINDOW_TITLE, Renderer, logo_images

CELL_TYPES = {"quad": pv.CellType.QUAD, "tri": pv.CellType.TRIANGLE}


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
                    clim=sc.clim, show_edges=sc.show_edges, edge_color=th.edges,
                    line_width=0.5, interpolate_before_map=True, n_colors=self.n_bands or 256,
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

        pl.add_text(sc.title, position="upper_left", font_size=12, color=th.foreground)
        self.status = pl.add_text("", position="lower_left", font_size=10, color=th.foreground)
        pl.add_text(BRAND_TEXT, position="lower_right", font_size=10, color=th.foreground)

        if sc.slider:
            label, vmin, vmax, cb_fn = sc.slider
            pl.add_slider_widget(cb_fn, rng=[vmin, vmax], value=vmin, title=label,
                                 pointa=(0.30, 0.90), pointb=(0.70, 0.90), color=th.foreground,
                                 style="modern")

        self._add_logos(pl, th)
        pl.view_xy()
        pl.camera.parallel_projection = True
        pl.reset_camera()
        pl.camera.zoom(1.15)
        pl.show(interactive_update=True, auto_close=False)
        self.pl = pl

    @staticmethod
    def _add_logos(pl, th, height=0.07, right=0.99, top=0.985, gap=0.01):
        """Place logos right-aligned in the top-right corner (viewport coordinates)."""
        ww, wh = pl.window_size
        x = right
        for path, aspect in reversed(logo_images(th)):
            w = height * aspect * wh / ww
            x -= w
            pl.add_logo_widget(path, position=(x, top - height), size=(w, height), opacity=1.0)
            x -= gap

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
        self.pl.update(stime=1, force_redraw=True)

    def _set_status(self, text):
        st = self.status
        if hasattr(st, "SetText") and hasattr(st, "GetText") and st.GetClassName() == "vtkCornerAnnotation":
            st.SetText(0, text)          # 0 = lower left corner
        else:
            st.SetInput(text)

    def is_open(self):
        pl = getattr(self, "pl", None)
        return pl is not None and not getattr(pl, "_closed", False) and pl.render_window is not None

    def close(self):
        if self.is_open():
            self.pl.close()
