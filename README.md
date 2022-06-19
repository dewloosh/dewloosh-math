[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dewloosh/dewloosh-core/main?labpath=examples%2Flpp.ipynb?urlpath=lab)
[![CircleCI](https://circleci.com/gh/dewloosh/dewloosh-math.svg?style=shield)](https://circleci.com/gh/dewloosh/dewloosh-math) 
[![Documentation Status](https://readthedocs.org/projects/dewloosh-math/badge/?version=latest)](https://nddict.readthedocs.io/en/latest/?badge=latest) 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/dewloosh.math.svg)](https://pypi.org/project/dewloosh.math) 

# **dewloosh.math**

> **Warning**
> This package is under active development and in an **alpha stage**. Come back later, or star the repo to make sure you don’t miss the first stable release!

This package contains common developer utilities to support other `dewloosh` solutions. Everything is pure Python, the package requires no extra dependencies and should run on a minimal setup.

The most important features:

* Various dictionary classes that enhance the core behaviour of the built-in `dict` type. The top of the cake is the `DeepDict` class, which offers a different behaviour for nested dictionaries by applying a self replicating defalt factory.

* A set of tools for metaprogramming. The use cases include declaring custom abstract class properties, using metaclasses to avoid unwanted code conflicts, assuring the implementation of abstract methods at design time, etc.

* Decorators, wrappers and other handy developer tools.

## **Documentation**

Click [here](https://dewloosh-math.readthedocs.io/en/latest/) to read the documentation.

## **Installation**
This is optional, but we suggest you to create a dedicated virtual enviroment at all times to avoid conflicts with your other projects. Create a folder, open a command shell in that folder and use the following command

```console
>>> python -m venv venv_name
```

Once the enviroment is created, activate it via typing

```console
>>> .\venv_name\Scripts\activate
```

`dewloosh.math` can be installed (either in a virtual enviroment or globally) from PyPI using `pip` on Python >= 3.6:

```console
>>> pip install dewloosh.math
```

## **Crash Course**

### Linear Programming

Example for unique solution
(0, 6, 0, 4) --> 10
The following order automatically creates
a feasble solution : [0, 2, 3, 1]

```python
>>> from dewloosh.math.function import Function, Equality, InEquality
>>> from dewloosh.math.optimize import LinearProgrammingProblem as LPP, \
>>>     DegenerateProblemError, NoSolutionError, BinaryGeneticAlgorithm
>>> variables = ['x1', 'x2', 'x3', 'x4']
>>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
>>> obj1 = Function(3*x1 + 9*x3 + x2 + x4, variables=syms)
>>> eq11 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
>>> eq12 = Equality(x2 + x3 - x4 - 2, variables=syms)
>>> lpp = LPP(cost=obj1, constraints=[eq11, eq12], variables=syms)
>>> lpp.solve(order=[0, 2, 3, 1], raise_errors=True)['x']
array([0., 6., 0., 4.])
```

## **Testing**

To run all tests, open up a console in the root directory of the project and type the following

```console
>>> python -m unittest
```

## **Dependencies**

* `Numba`, `NumPy`, `SciPy`, `SymPy`, `awkward`

## **License**

This package is licensed under the MIT license.