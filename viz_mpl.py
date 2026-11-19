"""matplotlib renderer: one PolyCollection updated in place.

Constant per-cell colouring only; the smooth option needs nodal interpolation
and is provided by the PyVista renderer.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider

import matplotlib.image as mpimg

from viz_base import BRAND_TEXT, WINDOW_TITLE, Renderer, logo_images


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
                                   edgecolors=th.edges, linewidths=0.3)
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

        self._add_logos(fig, th)
        self.fig, self.ax = fig, ax
        plt.show(block=False)
        fig.canvas.draw()

    @staticmethod
    def _add_logos(fig, th, height=0.07, right=0.985, top=0.985, gap=0.012):
        """Place logos right-aligned in the top-right corner (figure coordinates)."""
        fw, fh = fig.get_size_inches()
        x = right
        for path, aspect, scale in reversed(logo_images(th)):
            lh = height * scale
            w = lh * aspect * fh / fw
            x -= w
            lax = fig.add_axes([x, top - lh, w, lh])
            lax.imshow(mpimg.imread(path))
            lax.axis("off")
            x -= gap

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
        return plt.fignum_exists(self.fig.number)

    def close(self):
        plt.close(self.fig)
