"""Geometry base class and registry.

To add a new specimen, create a module in this package with a subclass of
:class:`Geometry` decorated by ``@register("name")`` and import it in
``geometries/__init__.py``.  Nothing outside this package needs to change.
"""
from abc import ABC, abstractmethod

GEOMETRIES = {}


def register(name):
    """Class decorator adding a Geometry subclass to the registry."""
    def deco(cls):
        cls.name = name
        GEOMETRIES[name] = cls
        return cls
    return deco


class Geometry(ABC):
    name = "abstract"
    #: default parameter values; keys define the accepted parameters
    default_params = {}
    #: help text for each parameter (optional)
    param_help = {}

    def params(self, **overrides):
        unknown = set(overrides) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; "
                             f"accepted: {sorted(self.default_params)}")
        p = dict(self.default_params)
        for k, v in overrides.items():
            # coerce to the type of the default (CLI passes strings)
            p[k] = type(self.default_params[k])(v)
        return p

    @abstractmethod
    def build(self, h, **params):
        """Return a spec.ProblemSpec meshed with element size ~h."""

    @classmethod
    def describe(cls):
        lines = [f"{cls.name}: {(cls.__doc__ or '').strip().splitlines()[0] if cls.__doc__ else ''}"]
        for k, v in cls.default_params.items():
            lines.append(f"    {k} = {v}   {cls.param_help.get(k, '')}")
        return "\n".join(lines)
