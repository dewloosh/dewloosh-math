[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dewloosh/dewloosh-core/main?labpath=examples%2Flpp.ipynb?urlpath=lab)
[![CircleCI](https://circleci.com/gh/dewloosh/dewloosh-math.svg?style=shield)](https://circleci.com/gh/dewloosh/dewloosh-math) 
[![Documentation Status](https://readthedocs.org/projects/dewloosh-math/badge/?version=latest)](https://dewloosh-math.readthedocs.io/en/latest/?badge=latest) 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/dewloosh.math.svg)](https://pypi.org/project/dewloosh.math) 

# **dewloosh.math**

> **Warning**
> This package is under active development and in an **beta stage**. Come back later, or star the repo to make sure you don’t miss the first stable release!

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

### Linear Algebra

Define a reference frame (B) relative to the ambient frame (A):
```python
>>> from dewloosh.math.linalg import ReferenceFrame
>>> A = ReferenceFrame(name='A', axes=np.eye(3))
>>> B = A.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ', name='B')
```
Get the DCM matrix of the transformation between two frames:
```python
>>> B.dcm(target=A)
```
Define a vector in frame A and view the components of it in frame B:
```python
>>> v = Vector([0.0, 1.0, 0.0], frame=A)
>>> v.view(B)
```
Define the same vector in frame B:
```python
>>> v = Vector(v.show(B), frame=B)
>>> v.show(A)
```

### Linear Programming

Solve a following Linear Programming Problem (LPP) with one 
unique solution:

```python
>>> from dewloosh.math.optimize import LinearProgrammingProblem as LPP
>>> import sympy as sy
>>> variables = ['x1', 'x2', 'x3', 'x4']
>>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
>>> obj1 = Function(3*x1 + 9*x3 + x2 + x4, variables=syms)
>>> eq11 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
>>> eq12 = Equality(x2 + x3 - x4 - 2, variables=syms)
>>> problem = LPP(cost=obj1, constraints=[eq11, eq12], variables=syms)
>>> problem.solve()['x']
array([0., 6., 0., 4.])
```

### NonLinear Programming

Find the minimizer of the Rosenbrock function:

```python
>>> from dewloosh.math.optimize import BinaryGeneticAlgorithm
>>> def Rosenbrock(x, y):
>>>     a = 1, b = 100
>>>     return (a-x)**2 + b*(y-x**2)**2
>>> ranges = [[-10, 10],[-10, 10]]
>>> BGA = BinaryGeneticAlgorithm(Rosenbrock, ranges, length=12, nPop=200)
>>> BGA.solve()
array([0.99389553, 0.98901176]) 
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
  * `networkx`

## **License**

This package is licensed under the MIT license.