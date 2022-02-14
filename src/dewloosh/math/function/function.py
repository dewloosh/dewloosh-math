# -*- coding: utf-8 -*-
from typing import TypeVar, Callable
import sympy as sy
from sympy import Expr, degree, symbols, latex, lambdify
from sympy.core.numbers import One
import numpy as np
from collections import OrderedDict

from dewloosh.core.tools import getasany

from .meta import MetaFunction, substitute


__all__ = ['Function', 'VariableManager', 'FuncionLike']


FuncionLike = TypeVar('FuncionLike', str, Callable, Expr)


class Function(MetaFunction):
    """
    Base class function.
    Initializes a base class for further operation.
    Possible arguments for **kwargs:
        value
        gradient
        Hessian
        domain
    """

    def __init__(self, f0: FuncionLike = None, f1: Callable = None,
                 f2: Callable = None, *args, **kwargs):
        super().__init__()
        self.update(f0, f1, f2, *args, **kwargs)

    def update(self, f0: FuncionLike = None, f1: Callable = None,
               f2: Callable = None, *args, **kwargs):
        self.from_str = None
        if f0 is not None:
            if isinstance(f0, str):
                kwargs.update(self._str_to_func(f0, **kwargs))
                self.from_str = True
            elif isinstance(f0, Expr):
                kwargs.update(self._sympy_to_func(f0, **kwargs))

        self.expr = kwargs.get('expr', None)
        self.variables = kwargs.get('variables', None)
        self.f0 = kwargs.get('value', f0)
        self.f1 = kwargs.get('gradient', f1)
        self.f2 = kwargs.get('Hessian', f2)
        self.dimension = getasany(['d', 'dimension', 'dim'], None, **kwargs)
        self.domain = kwargs.get('domain', None)
        self.vmap = kwargs.get('vmap', None)

    @property
    def symbolic(self):
        try:
            return self.expr is not None
        except AttributeError:
            return False

    @property
    def linear(self):
        if self.symbolic:
            return all(np.array([degree(self.expr, v)
                                 for v in self.variables], dtype=int) == 1)
        else:
            try:
                G = self.G().astype(np.int32)
                assert np.all((G == 0))
                return True
            except AssertionError:
                return False

    def linear_coefficients(self, normalize=False):
        d = self.coefficients(normalize)
        if d:
            return {key: value for key, value in d.items()
                    if len(key.free_symbols) <= 1}
        return None

    def coefficients(self, normalize=False):
        try:
            d = OrderedDict({x: 0 for x in self.variables})
            d.update(self.expr.as_coefficients_dict())
            if not normalize:
                return d
            else:
                res = OrderedDict()
                for key, value in d.items():
                    if len(key.free_symbols) == 0:
                        res[One()] = value*key
                    else:
                        res[key] = value
                return res
        except Exception:
            return None

    def to_latex(self):
        try:
            return latex(self.expr)
        except Exception:
            return None

    def subs(self, values, variables=None, inplace=False):
        if self.expr is None:
            return None
        expr = substitute(self.expr, values, variables,
                          as_string=self.from_str)
        kwargs = self._sympy_to_func(expr=expr, variables=variables)
        if not inplace:
            return Function(None, None, None, **kwargs)
        else:
            self.update(None, None, None, **kwargs)
            return self


class VariableManager(object):

    def __init__(self, variables=None, **kwargs):
        try:
            variables = list(sy.symbols(variables, **kwargs))
        except Exception:
            variables = variables
        try:
            self.vmap = OrderedDict({v: v for v in variables})
        except Exception:
            self.vmap = OrderedDict()

    def substitute(self, vmap: dict = None, inverse=False):
        if not inverse:
            sval = list(vmap.values())
            svar = list(vmap.keys())
        else:
            sval = list(vmap.keys())
            svar = list(vmap.values())
        for v, expr in self.vmap.items():
            self.vmap[v] = substitute(expr, sval, svar)

    def lambdify(self, variables=None):
        assert variables is not None
        for v, expr in self.vmap.items():
            self.vmap[v] = lambdify([variables], expr, 'numpy')

    def __call__(self, v):
        return self.vmap[v]

    def target(self):
        return list(self.vmap.keys())

    def source(self):
        s = set()
        for expr in self.vmap.values():
            s.update(expr.free_symbols)
        return list(s)

    def add_variables(self, variables, overwrite=True):
        if overwrite:
            self.vmap.update({v: v for v in variables})
        else:
            for v in variables:
                if v not in self.vmap:
                    self.vmap[v] = v


if __name__ == '__main__':

    def f0(x=None, y=None):
        return x**2 + y

    def f1(x=None, y=None):
        return np.array([2*x, 1])

    def f2(x=None, y=None):
        return np.array([[0, 0], [0, 0]])

    f = Function(f0, f1, f2, d=2)
    print(f.linear)

    g = Function('3*x + 4*y - 2', variables=['x', 'y', 'z'])
    print(g([1, 2, 30]))

    h = Function('3*x + 4*y - 2')
    print(h([1, 2]))

    m = Function('3*x + 4*y - 2', variables=symbols('x y z'))
    print(m([1, 2, -30]))

    x1, x3 = sy.symbols(['x1', 'x3'], positive=False)
    x2, x4 = sy.symbols(['x2', 'x4'], positive=True)
    variables = [x1, x2, x3, x4]
    mngr = VariableManager(variables)
    y1, y2 = sy.symbols(['y1', 'y2'])
    mngr.substitute({y1: 1, y2: 2})
