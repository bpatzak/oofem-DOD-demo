"""matplotlib renderer: one PolyCollection updated in place.

Constant per-cell colouring only; the smooth option needs nodal interpolation
and is provided by the PyVista renderer.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider

import matplotlib.image as mpimg
from matplotlib.colors import to_rgba

from viz_base import BRAND_TEXT, QUIT_KEYS, WINDOW_TITLE, Renderer, logo_images


class MatplotlibRenderer(Renderer):
    def setup(self, sc):
        if sc.smooth:
            print("matplotlib backend: --smooth not supported, using cell values")
        th = sc.theme
        self.sc = sc
        plt.ion()
        fig = plt.figure(figsize=(14, 4.8), facecolor=th.background)
        try:
            fig.canvas.manager.set_window_title(WINDOW_TITLE)
        except AttributeError:
            pass
        bottom = 0.22 if sc.slider else 0.12
        ax = fig.add_axes([0.04, bottom, 0.84, 0.72], facecolor=th.background)
        cax = fig.add_axes([0.90, bottom, 0.015, 0.72])

        xy = sc.coords
        ax.fill(*self._outline(xy), facecolor="none", edgecolor=th.outline, lw=0.8, ls="--")
        self.coll = PolyCollection(xy[sc.cells], array=np.zeros(len(sc.cells)),
                                   cmap=th.cmap_signed if sc.signed else th.cmap_positive,
                                   edgecolors=to_rgba(th.edges, sc.mesh_alpha) if sc.show_mesh else "none",
                                   linewidths=0.3 if sc.show_mesh else 0.0)
        self.coll.set_clim(*sc.clim)
        ax.add_collection(self.coll)

        sup = sc.groups.get("supports")
        if sup is not None and len(sup):
            ax.plot(xy[sup, 0], xy[sup, 1], marker="^", ls="none", ms=12,
                    color=th.support, transform=ax.transData, clip_on=False)
        self.load_ids = sc.groups.get("load")
        (self.load_mark,) = ax.plot([], [], marker="v", ls="none", ms=12, color=th.load)

        m = sc.warp_margin
        ax.set_xlim(xy[:, 0].min() - 0.03 * np.ptp(xy[:, 0]), xy[:, 0].max() + 0.03 * np.ptp(xy[:, 0]))
        ax.set_ylim(xy[:, 1].min() - m - 2.0, xy[:, 1].max() + 0.5 * m + 4.0)
        ax.set_aspect("equal")
        ax.tick_params(colors=th.foreground)
        for s in ax.spines.values():
            s.set_color(th.outline)

        cb = fig.colorbar(self.coll, cax=cax)
        cb.set_label(sc.field_label, color=th.foreground)
        cb.ax.tick_params(colors=th.foreground)
        cb.outline.set_edgecolor(th.outline)

        ax.set_title(sc.title, color=th.foreground, loc="left")
        self.status = fig.text(0.04, 0.955, "", color=th.foreground, family="monospace", va="top")
        fig.text(0.985, 0.015, BRAND_TEXT, color=th.foreground, ha="right", va="bottom",
                 fontsize=10, alpha=0.8)

        if sc.slider:
            label, vmin, vmax, cb_fn = sc.slider
            sax = fig.add_axes([0.12, 0.05, 0.6, 0.04], facecolor=th.background)
            self.slider = Slider(sax, label, vmin, vmax, valinit=vmin, color=th.load)
            self.slider.label.set_color(th.foreground)
            self.slider.valtext.set_color(th.foreground)
            self.slider.on_changed(cb_fn)

        self.fig, self.ax = fig, ax
        self._logos = []
        for path, aspect, scale in logo_images(th):
            lax = fig.add_axes([0, 0, 0.01, 0.01])
            lax.imshow(mpimg.imread(path))
            lax.axis("off")
            self._logos.append((lax, aspect, scale))
        self._place_logos()
        # figure-fraction positions depend on the window size: redo on resize
        fig.canvas.mpl_connect("resize_event", lambda event: self._place_logos())
        # quit keys, independent of matplotlib's (configurable) default key map
        self._quit = False
        quit_keys = {k.lower() for k in QUIT_KEYS}
        fig.canvas.mpl_connect("key_press_event",
                               lambda event: setattr(self, "_quit", self._quit or
                                                     (event.key or "").lower() in quit_keys))
        plt.show(block=False)
        fig.canvas.draw()

    def _place_logos(self, height_px=34, right_px=12, top_px=6, gap_px=14, max_width=0.38):
        """Right-align logos in the top-right corner with fixed pixel sizes.

        Converted to figure fractions for the current window size, so the
        logos keep their size and corner position when the window is resized;
        in narrow windows they shrink to at most max_width of the width.
        """
        ww, wh = self.fig.canvas.get_width_height()
        if ww <= 0 or wh <= 0:
            return
        total = sum(height_px * s * a for _, a, s in self._logos) + gap_px * (len(self._logos) - 1)
        f = min(1.0, max_width * ww / total) if total > 0 else 1.0
        x = ww - right_px
        for lax, aspect, scale in reversed(self._logos):
            h = height_px * scale * f
            w = h * aspect
            x -= w
            lax.set_position([x / ww, (wh - top_px - h) / wh, w / ww, h / wh])
            x -= gap_px * f

    @staticmethod
    def _outline(xy):
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        return [xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]

    def update(self, def_coords, cell_values, node_values, status):
        self.coll.set_verts(def_coords[self.sc.cells])
        self.coll.set_array(cell_values)
        if self.load_ids is not None:
            p = def_coords[self.load_ids]
            self.load_mark.set_data(p[:, 0], p[:, 1] + 1.0)
        self.status.set_text(status)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def is_open(self):
        return not self._quit and plt.fignum_exists(self.fig.number)

    def close(self):
        plt.close(self.fig)
