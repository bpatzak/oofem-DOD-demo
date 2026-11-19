# Real-time beam demo (OOFEM)

Interactive FE model of a specimen loaded by a prescribed displacement coming
from an experimental rig in real time. Built on [OOFEM](https://www.oofem.org)
via its Python bindings.

The model is set up once. For every new input value, the loop updates the
prescribed boundary condition and runs one OOFEM `LinearStatic` step. The
stiffness matrix is assembled and factorized in the first step only, so later
steps are just a right-hand side plus a back-substitution. The deformed mesh is
then redrawn, coloured by the selected stress or strain field.

## Setup
```
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install pyvista
export OOFEM_BUILD=/path/to/oofem/releaseBuild   # dir containing oofempy*.so (use a Release build)
export OOFEM_SRC=/path/to/oofem                    # source tree (bindings/python/util.py)
```

## Usage
```
python demo.py                                   # 3-point bending, PyVista, sine input
python demo.py --backend mpl --theme dark --field mises
python demo.py --h 1 --smooth --bands 12         # nodal values, banded contours
python demo.py --source manual --max 8           # slider drives the displacement
python demo.py --geometry beam4pb --param load_span=80
python demo.py --bench 200                       # solve timings for several mesh sizes
python demo.py --list                            # geometries, parameters, fields
```
`--max` is the largest expected prescribed displacement. The colour scale is
fixed from it, using the unit solution × max (exact for the linear model), so
the stress builds up gradually within a stable scale. Timing statistics per
loop phase are printed every `--report` frames.

## Structure
| File | Role |
|---|---|
| `geometries/` | Specimens. Each one builds a neutral `ProblemSpec`: mesh, fixed DOFs, control channels |
| `spec.py` | `ProblemSpec` and `Result` data contracts |
| `fe_model.py` | Generic OOFEM model built from a `ProblemSpec`, with `solve({channel: value})` |
| `fields.py` | Derived scalar fields (σxx, von Mises, εxx, ...) and fixed colour limits |
| `sources.py` | Input sources (sine, ramp, manual slider); the real device goes here |
| `viz_base.py` | Renderer interface, light and dark themes, branding, logos |
| `viz_pv.py`, `viz_mpl.py` | PyVista and matplotlib renderers |
| `demo.py` | Main loop, command-line options, benchmark |

### Adding a geometry
Create `geometries/<name>.py` with a `Geometry` subclass decorated by
`@register("<name>")`. Its `build(h, **params)` returns a `ProblemSpec`. Then
import the module in `geometries/__init__.py`. See `beam4pb.py`. The helpers in
`mesh_utils.py` generate structured grids whose nodes fall exactly on given
breakpoints (supports, load points).

### Adding an input source
Subclass `sources.Source` and implement a non-blocking `read()` that returns
the latest `{channel: value}`.

### Logos
Logos are shown in the top-right corner when their files exist in `assets/`.
The list is `LOGOS` in `viz_base.py`; each entry can have a separate dark-theme
variant:
- `oofem_logo.png`: the OOFEM wordmark, used for both themes
- `fsv_logo.png`, `fsv_logo_dark.png`: Faculty of Civil Engineering, CTU in
  Prague, made transparent from the official logo. The dark variant has white
  lettering. Both have a small transparent margin, which avoids edge artefacts
  in VTK.

The size of each logo can be adjusted with the `scale` key in `LOGOS`. The
caption at the bottom right is `BRAND_TEXT` in `viz_base.py`.
