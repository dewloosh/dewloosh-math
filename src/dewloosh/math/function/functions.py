# -*- coding: utf-8 -*-
import sympy as sy
from dewloosh.math.function.testfunction import TestMinFunction
from dewloosh.math.function.meta import symbolize, substitute


def Rosenbrock(a=1, b=100):
    """
    Creates the Rosenbrock function object.

    f(x,y) = (a-x)**2 + b*(y-x**2)**2

    a,b are constants, tipically a is set to 1 and  b is set to 100.

    The global minimum
    f(1,1) = 0.
    """
    str_expr = '(a-x)**2 + b*(y-x**2)**2'
    expr = substitute(sy.sympify(str_expr), [a, b], ['a', 'b'])
    return TestMinFunction(**symbolize(expr=expr), optimums=[(1.0, 1.0)])


def Himmelblau():
    """
    Creates the Himmelblau's function object.

    f(x,y) = (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    It has four identical local minima with values 0.0 at
    (3.0,2.0),
    (-2.805,3.131),
    (-3.779,-3.283),
    (3.584,-1.848).
    """
    str_expr = '(x**2 + y - 11)**2 + (x + y**2 - 7)**2'
    return TestMinFunction(**symbolize(str_expr=str_expr),
                           optimums=[(3.0, 2.0), (-2.805118, 3.131312),
                                     (-3.779310, -3.283186),
                                     (3.584428, -1.848126)])


def GoldsteinPrice():
    """
    Creates the Goldstein-Price function object.

    f(x,y) = (1+(x+y+1)**2 * (19-14*x+3*x**2-14*y+6*x*y+3*y**2))
    *(30+(2*x-3*y)**2 * (18-32*x+12*x**2+48*y-36*x*y+27*y**2))

    The global minimum
    f(0,-1) = 3.
    """
    str_expr = '(1+(x+y+1)**2 * (19-14*x+3*x**2-14*y+6*x*y+3*y**2))' \
        '*(30+(2*x-3*y)**2 * (18-32*x+12*x**2+48*y-36*x*y+27*y**2))'
    return TestMinFunction(**symbolize(str_expr=str_expr),
                           optimums=[(0.0, -1.0)])


def Beale():
    """
    Creates the Beale function object.

    f(x,y) = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2
    + (2.625 - x + x*y**3)**2

    The global minimum
    f(3,0.5) = 0.
    """
    str_expr = '(1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 ' \
        '+ (2.625 - x + x*y**3)**2'
    return TestMinFunction(**symbolize(str_expr=str_expr),
                           optimums=[(3.0, 0.5)])


def Matyas():
    """
    Creates the Matyas function object.

    f(x, y) = 0.26*(x**2 + y**2) - 0.48*x*y

    The global minimum
    f(0.,0.) = 0.
    """
    str_expr = '0.26*(x**2 + y**2) - 0.48*x*y'
    return TestMinFunction(**symbolize(str_expr=str_expr),
                           optimums=[(0.0, 0.0)])


if __name__ == '__main__':

    """
    check:
        - Himmelblau with initial = [-1.,-1.]. Problem with the Hessian
          at [-0.270845, -0.923039].

    """
