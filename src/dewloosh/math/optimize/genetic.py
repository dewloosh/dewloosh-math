# -*- coding: utf-8 -*-
from abc import abstractmethod
import numpy as np


__all__ = ['BinaryGeneticAlgorithm']


def even(n): return n % 2 == 0
def odd(n): return not even(n)


class GeneticAlgorithm:
    """
    Binary Generic algorithm for continuous problems.

    positional arguments:
        fnc : Callable
        ranges : sequence of pairs of limits for each variable
        length : chromosome length / string length
        p_c : probability of crossover
        p_m : probability of mutation
        nPop : number of members in the population
        elitism : float or integer
    """

    def __init__(self, fnc, ranges, length=5, p_c=1, p_m=0.2, nPop=100,
                 *args, **kwargs):
        super().__init__()
        self.fnc = fnc
        self.ranges = np.array(ranges)
        self.dim = fnc.dimension
        self.length = length
        self.p_c = p_c
        self.p_m = p_m

        # Second half of the population is used as a pool to make parents.
        # This assumes that population size is a multiple of 4.
        if odd(nPop):
            nPop += 1
        if odd(int(nPop/2)):
            nPop += 2
        assert nPop % 4 == 0
        assert nPop >= 4

        self.nPop = nPop
        self._genotypes = None
        self._fittness = None
        self.reset()
        self.set_solution_params(**kwargs)

    @property
    def genotypes(self):
        return self._genotypes

    @genotypes.setter
    def genotypes(self, value):
        self._genotypes = value
        self.phenotypes = self.decode(self._genotypes)

    def reset(self):
        self._evolver = self.evolver()
        self._evolver.send(None)

    def set_solution_params(self, tol=1e-12, maxiter=200, miniter=100,
                            elitism=1, **kwargs):
        self.tol = tol
        self.maxiter = np.max([miniter, maxiter])
        self.miniter = np.min([miniter, maxiter])
        self.elitism = elitism

    def evolver(self):
        self.genotypes = self.populate()
        _ = yield
        while True:
            self.genotypes = self.populate(
                self.select(self._genotypes, self.phenotypes))
            yield self._genotypes

    def evolve(self, cycles=1):
        for _ in range(cycles):
            next(self._evolver)
        return self.genotypes

    def criteria(self) -> bool:
        value = yield
        while True:
            _value = yield
            yield abs(value - _value) < self.tol
            value = _value

    def solve(self, reset=False, returnlast=False, **kwargs):
        if reset:
            self.reset()
        self.set_solution_params(**kwargs)
        criteria = self.criteria()
        criteria.send(None)
        criteria.send(self.fnc(self.best_phenotype()))
        finished = False
        nIter = 0
        while (not finished and nIter < self.maxiter) or \
                (nIter < self.miniter):
            next(self._evolver)
            finished = criteria.send(self.fnc(self.best_phenotype()))
            next(criteria)
            nIter += 1
        self.nIter = nIter
        return self.best_phenotype(lastknown=returnlast)

    def fittness(self, phenotypes=None, dtype=np.float32):
        if phenotypes is not None:
            self._fittness = np.array([self.fnc(x) for x in phenotypes],
                                      dtype=dtype)
        return self._fittness.astype(dtype)

    def best_phenotype(self, lastknown=False):
        if lastknown:
            fittness = self._fittness
        else:
            fittness = self.fittness(self.phenotypes)
        best = np.argmin(fittness)
        return self.phenotypes[best]

    def divide(self, fittness=None):
        """
        Divides population to elit and others,
        and returns the corresponding index arrays.
        """
        if self.elitism < 1:
            argsort = np.argsort(fittness)
            elit = argsort[:int(self.nPop*self.elitism)]
            others = argsort[int(self.nPop*self.elitism):]
        elif self.elitism > 1.1:
            argsort = np.argsort(fittness)
            elit = argsort[:self.elitism]
            others = argsort[self.elitism:]
        else:
            elit = []
            others = list(range(self.nPop))
        return list(elit), others

    def random_parents_generator(self, genotypes=None):
        """
        Returns random pairs from a list of genotypes.
        This assumes theat the length of the input array is
        a multiple of 2.
        """
        n = len(genotypes)
        assert n % 2 == 0
        pool = np.full(n, True)
        nPool = n
        while nPool > 2:
            where = np.argwhere(pool == True).flatten()
            nPool = len(where)
            pair = np.random.choice(where, 2, replace=False)
            parent1 = genotypes[pair[0]]
            parent2 = genotypes[pair[1]]
            pool[pair] = False
            yield parent1, parent2

    @abstractmethod
    def populate(self, genotypes=None):
        ...

    @abstractmethod
    def decode(self, genotypes=None):
        ...

    @abstractmethod
    def crossover(self, parent1=None, parent2=None):
        ...

    @abstractmethod
    def mutate(self, child=None):
        ...

    @abstractmethod
    def select(self, genotypes=None, phenotypes=None):
        ...


class BinaryGeneticAlgorithm(GeneticAlgorithm):

    def populate(self, genotypes=None):
        nPop = self.nPop
        if genotypes is None:
            poolshape = (int(nPop / 2), self.dim * self.length)
            genotypes = np.random.randint(2, size=poolshape)
        else:
            poolshape = genotypes.shape
        nParent = poolshape[0]
        if nParent < nPop:
            offspring = []
            g = self.random_parents_generator(genotypes)
            try:
                while (len(offspring) + nParent) < nPop:
                    parent1, parent2 = next(g)
                    offspring.extend(self.crossover(parent1, parent2))
                genotypes = np.vstack([genotypes, offspring])
            except Exception:
                raise RuntimeError
        return genotypes

    def decode(self, genotypes=None):
        span = (2**self.length - 2**0)
        genotypes = genotypes.reshape((self.nPop, self.dim, self.length))
        precisions = [(self.ranges[d, -1] - self.ranges[d, 0]) / span
                      for d in range(self.dim)]
        phenotypes = \
            np.sum([genotypes[:, :, i]*2**i
                    for i in range(self.length)], axis=0).astype(np.float32)
        for d in range(self.dim):
            phenotypes[:, d] *= precisions[d]
            phenotypes[:, d] += self.ranges[d, 0]
        return phenotypes

    def crossover(self, parent1=None, parent2=None, nCut=None):
        if np.random.rand() > self.p_c:
            return parent1, parent2

        if nCut is None:
            nCut = np.random.randint(1, self.dim*self.length-1)

        cuts = [0, self.dim * self.length]
        p = np.random.choice(range(1, self.length * self.dim - 1),
                             nCut, replace=False)
        cuts.extend(p)
        cuts = np.sort(cuts)

        child1 = np.zeros(self.dim*self.length, dtype=np.int32)
        child2 = np.zeros(self.dim*self.length, dtype=np.int32)

        randBool = np.random.rand() > 0.5
        for i in range(nCut+1):
            if (i % 2 == 0) == randBool:
                child1[cuts[i]:cuts[i+1]] = parent1[cuts[i]:cuts[i+1]]
                child2[cuts[i]:cuts[i+1]] = parent2[cuts[i]:cuts[i+1]]
            else:
                child1[cuts[i]:cuts[i+1]] = parent2[cuts[i]:cuts[i+1]]
                child2[cuts[i]:cuts[i+1]] = parent1[cuts[i]:cuts[i+1]]

        return self.mutate(child1), self.mutate(child2)

    def mutate(self, child=None):
        p = np.random.rand(self.dim*self.length)
        return np.where(p > self.p_m, child, 1-child)

    def select(self, genotypes=None, phenotypes=None):
        fittness = self.fittness(phenotypes)
        winners, others = self.divide(fittness)
        while len(winners) < int(self.nPop / 2):
            candidates = np.random.choice(others, 3, replace=False)
            winner = np.argsort([fittness[ID] for ID in candidates])[0]
            winners.append(candidates[winner])
        return np.array([genotypes[w] for w in winners], dtype=np.float32)


if __name__ == '__main__':

    def Rosenbrock(a, b, x, y):
        return (a-x)**2 + b*(y-x**2)**2

    def f(x):
        return Rosenbrock(1, 100, x[0], x[1])

    f.dimension = 2
    ranges = [
        [-10, 10],
        [-10, 10]
    ]
    BGA = BinaryGeneticAlgorithm(f, ranges, length=12, nPop=200)
    r = BGA.solve()
    print(r)
    
    # plot the history
    import matplotlib.pyplot as plt
    BGA = BinaryGeneticAlgorithm(f, ranges, length=12, nPop=200)
    history = [f(BGA.best_phenotype())]
    for _ in range(100):
        BGA.evolve(1)
        history.append(f(BGA.best_phenotype()))
    plt.plot(history)
    plt.show()
    x = BGA.best_phenotype()
    fx = f(x)
    print('min {} @ {}'.format(fx,x))
    
