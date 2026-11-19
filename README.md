# Real-time beam demo (OOFEM)

Interactive FE model of a specimen loaded by a prescribed displacement coming
from an experimental rig in real time. Built on [OOFEM](https://www.oofem.org)
via its Python bindings.

## Setup
```
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install pyvista
export OOFEM_BUILD=/path/to/oofem/releaseBuild   # dir containing oofempy*.so
export OOFEM_SRC=/path/to/oofem                    # source tree (bindings/python/util.py)
```
