=======================
Genetic Algorithms (GA)
=======================

The `GeneticAlgorithm` class provides a skeleton for the implementation of
custom GAs, and the `BinaryGeneticAlgorithm` is the standard implementation of it.

For a good explanation of how Genetic Algorithms work, read 
`this <https://www.mathworks.com/help/gads/how-the-genetic-algorithm-works.html>`_
from 
`MathWorks <https://www.mathworks.com/?s_tid=gn_logo>`_.

.. autoclass:: dewloosh.math.optimize.GeneticAlgorithm
    :members: solve, populate, decode, mutate, crossover, select

.. autoclass:: dewloosh.math.optimize.BinaryGeneticAlgorithm
    :members: populate, decode, mutate, crossover, select
