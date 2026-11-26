# Real-time beam demo (OOFEM)

Interactive FE model of a specimen loaded by a prescribed displacement coming
from an experimental rig in real time. Built on [OOFEM](https://www.oofem.org)
via its Python bindings.

The model is set up once. For every new input value, the loop updates the
prescribed boundary condition and runs one OOFEM `LinearStatic` step. The
stiffness matrix is assembled and factorized in the first step only, so later
steps are just a right-hand side plus a back-substitution. The deformed mesh is
then redrawn, coloured by the selected stress or strain field.

## Prerequisites
- **OOFEM built with Python bindings.** Configure OOFEM with
  `USE_PYBIND_BINDINGS=ON` (and the default `USE_SM=ON`) to get the `oofempy`
  module. A **Release build is recommended**: in a Debug build the solve is
  much slower, and the loop will not run in real time.
  ```
  cd /path/to/oofem
  cmake -S . -B releaseBuild -DCMAKE_BUILD_TYPE=Release -DUSE_PYBIND_BINDINGS=ON
  cmake --build releaseBuild --target oofempy -j8
  ```
  This needs the Python development headers (e.g. `python3-dev`). Build with
  the same Python interpreter that runs the demo, because the module only
  works with the Python version it was built for.
- Python 3 with numpy and matplotlib, plus PyVista for the default renderer.

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
python demo.py --skip-unchanged 0.001            # no re-solve while input moves < 0.001 mm
python demo.py --mesh-alpha 0.2                  # fainter element edges (--no-mesh hides them)
python demo.py --logo-height 40                  # logo size in pixels
python demo.py --param E=30000                   # Young's modulus [MPa] (any geometry parameter)
python demo.py --source telemetry --telemetry-ip 10.34.2.43 --skip-unchanged 0.001
python demo.py --source telemetry --autocalibrate   # E follows the measured force
```
The text lines on screen are labelled:
- `geometry:` the specimen and its dimensions,
- `model:` mesh size, number of DOFs and Young's modulus `E` of the model,
- `input:` the input source and its state (for telemetry: address, link state,
  measured force),
- `simulation:` the prescribed displacement, the corresponding reaction force
  `F` (the force the actuator applies, positive in the direction of the
  displacement), the maximum of the displayed field and, with
  `--autocalibrate`, the calibrated `E`,
- `stat:` the timings.

Press `q` or `Esc` in the window (or close it) to quit.

`--max` is the largest expected prescribed displacement. The colour scale is
fixed from it, using the unit solution × max (exact for the linear model), so
the stress builds up gradually within a stable scale. With `--grow-legend`,
the range is enlarged (with 10 % headroom) whenever the displayed field goes
beyond it, e.g. for an input above `--max` or a stiffer calibrated specimen;
it never shrinks. Timing statistics per loop phase are printed every
`--report` frames.

### Real rig (telemetry)
The experiment's telemetry bridge (`Experiment/server.py`) runs on the machine
PC and serves the current displacement and force at `GET /api/values`
(`"relative_mm,force_N"`). `--source telemetry` polls it in a background thread
(`--telemetry-rate`, default 15 Hz), so the FE loop never waits for the network.
```
python demo.py --source telemetry --telemetry-ip <machine PC address> --max <expected max mm>
```
- `--telemetry-url URL` instead of `--telemetry-ip` for another port or path.
- `--telemetry-scale -1` if the machine's positive displacement is opposite to
  the model's (for the beams, positive = downward deflection).
- When the machine stops sending (204) or the server is unreachable, the last
  displacement is held and the `input:` line says why and for how long; the
  source reconnects automatically.
- Telemetry changes in 0.01 mm steps, so `--skip-unchanged 0.001` avoids
  re-solving identical inputs.

The bridge needs `aiohttp`; `Experiment/receiver.py` is a standalone client
for checking the connection.

### Autocalibration of Young's modulus
With `--autocalibrate`, the simulated reaction is compared with the force
measured on the rig. For this linear, displacement-controlled model the
reaction is proportional to `E`, so the modulus that matches the measurement is
`E = E0 * |F_measured| / |F_simulated(E0)|`.
- Calibration starts when the force deviation exceeds `--autocalib-tol`
  (default 0.10 = 10 %). `E` then moves continuously towards the matching value,
  smoothed with time constant `--autocalib-tau` (default 1 s) to average out
  noise, and stops once the deviation is below 1 %.
- No calibration without a fresh measurement or while the simulated force is
  below `--autocalib-min` times the force at `--max` (default 0.1): near zero
  load the force ratio is dominated by noise.
- Force magnitudes are compared, so the sign convention of the rig's force
  channel does not matter.
- A new `E` needs no new solve: displacements and strains do not depend on
  `E`, and stresses and reactions scale with it (`Result.stiffness_scaled`).
  The colour range stays fixed; use `--grow-legend` to enlarge it when the
  calibrated stresses exceed it.

## Structure
| File | Role |
|---|---|
| `geometries/` | Specimens. Each one builds a neutral `ProblemSpec`: mesh, fixed DOFs, control channels |
| `spec.py` | `ProblemSpec` and `Result` data contracts |
| `fe_model.py` | Generic OOFEM model built from a `ProblemSpec`, with `solve({channel: value})` |
| `fields.py` | Derived scalar fields (σxx, von Mises, εxx, ...) and fixed colour limits |
| `sources.py` | Input sources (sine, ramp, manual slider, telemetry from the rig) |
| `calibration.py` | Autocalibration of Young's modulus from the measured force |
| `viz_base.py` | Renderer interface, light and dark themes, branding, logos |
| `viz_pv.py`, `viz_mpl.py` | PyVista and matplotlib renderers |
| `demo.py` | Main loop, command-line options, benchmark |
| `Experiment/` | Telemetry bridge on the machine PC (`server.py`) and a test client (`receiver.py`) |

### Adding a geometry
Create `geometries/<name>.py` with a `Geometry` subclass decorated by
`@register("<name>")`. Its `build(h, **params)` returns a `ProblemSpec`. Then
import the module in `geometries/__init__.py`. See `beam4pb.py`. The helpers in
`mesh_utils.py` generate structured grids whose nodes fall exactly on given
breakpoints (supports, load points).

### Adding an input source
Subclass `sources.Source` and implement a non-blocking `read()` that returns
the latest `{channel: value}`. `status()` returns the text of the `input:` line;
`measured()` returns the measured force `{channel: force}` (or `None`), which
the autocalibration uses.

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
