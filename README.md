[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dewloosh/dewloosh-core/main?labpath=examples%2Flpp.ipynb?urlpath=lab)
[![CircleCI](https://circleci.com/gh/dewloosh/dewloosh-math.svg?style=shield)](https://circleci.com/gh/dewloosh/dewloosh-math) 
[![Documentation Status](https://readthedocs.org/projects/dewloosh-math/badge/?version=latest)](https://nddict.readthedocs.io/en/latest/?badge=latest) 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/dewloosh.math.svg)](https://pypi.org/project/dewloosh.math) 

# **dewloosh.math**

> **Warning**
> This package is under active development and in an **alpha stage**. Come back later, or star the repo to make sure you don’t miss the first stable release!

`dewloosh.math` is a rapid prototyping platform focused on numerical calculations mainly corcerned with simulations of natural phenomena. It provides a set of common functionalities and interfaces with a number of state-of-the-art open source packages to combine their power seamlessly under a single development environment.

The most important features:

* Numba-jitted classes and an extendible factory to define and manipulate vectors and tensors.

* Classes to define and solve linear and nonlinear optimization problems.

* A set of array routines for fast prorotyping, including random data  creation to assure well posedness, or other properties of test problems.

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
>>> lpp.solve()['x']
array([0., 6., 0., 4.])

```

Solve a following Linear Programming Problem (LPP) with one 
unique solution:

$$
\begin{eqnarray}
    & minimize&  \quad  3 x_1 + x_2 + 9 x_3 + x_4  \\
    & subject \, to& & \\
    & & x_1 + 2 x_3 + x_4 \,=\, 4, \\
    & & x_2 + x_3 - x_4 \,=\, 2, \\
    & & x_i \,\geq\, \, 0, \qquad i=1, \ldots, 4.
\end{eqnarray}
$$

```python
>>> from dewloosh.math.optimize import LinearProgrammingProblem as LPP
>>> import sympy as sy
>>> variables = ['x1', 'x2', 'x3', 'x4']
>>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
>>> obj1 = Function(3*x1 + 9*x3 + x2 + x4, variables=syms)
>>> eq11 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
>>> eq12 = Equality(x2 + x3 - x4 - 2, variables=syms)
>>> problem = LPP(cost=obj1, constraints=[eq11, eq12], variables=syms)
>>> problem.solve()
array([0., 6., 0., 4.])
```

## **Testing**

To run all tests, open up a console in the root directory of the project and type the following

```console
>>> python -m unittest
```

## **Dependencies**

must have 
  * `Numba`, `NumPy`, `SciPy`, `SymPy`, `awkward`

optional 
  * `newtorkx`

## **License**

This package is licensed under the MIT license.