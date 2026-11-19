"""Locate and import the OOFEM python module (oofempy).

The build directory holding ``oofempy*.so`` is taken from (first match wins):
  1. explicit argument to :func:`import_oofem`,
  2. environment variable ``OOFEM_BUILD``,
  3. DEFAULT_BUILD below.
The OOFEM source tree's ``bindings/python`` (for ``util.py``) is derived from
``OOFEM_SRC`` or DEFAULT_SRC.

Several OOFEM checkouts may exist on a machine, each with its own oofempy.so;
importing one built from a different source tree than expected leads to
confusing binding mismatches, so the path is set explicitly here.
"""
import os
import sys

DEFAULT_SRC = "/home/bp/devel/bpatzak-oofem.git"
DEFAULT_BUILD = os.path.join(DEFAULT_SRC, "releaseBuild")


def import_oofem(build_dir=None):
    """Prepend the OOFEM build and bindings dirs to sys.path; return (oofempy, util)."""
    build_dir = build_dir or os.environ.get("OOFEM_BUILD", DEFAULT_BUILD)
    src_dir = os.environ.get("OOFEM_SRC", DEFAULT_SRC)
    for p in (os.path.join(src_dir, "bindings", "python"), build_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import oofempy  # noqa: E402
    import util  # noqa: E402  (oofem bindings/python/util.py)
    return oofempy, util
