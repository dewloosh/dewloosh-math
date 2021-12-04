# -*- coding: utf-8 -*-
import numpy as np
import sympy as sy
from dewloosh.math import Function, InEquality, Equality, VariableManager
from polydata.core.tools import getasany
from collections import defaultdict, Iterable
from dewloosh.math.function.relation import Relations, Relation
from dewloosh.math.function.meta import coefficients
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import LinAlgError
from copy import copy


class LLP(object):
    def __init__(self, *args, constraints = None, variables = None,
                 positive = True, **kwargs):
        super().__init__()
        self.standardform = False
        self.obj = getasany(['obj', 'cost', 'payoff', 'fittness', 'f'],
                            None, **kwargs)
        self.constraints = [] if constraints is None else constraints
        self.vmanager = VariableManager(variables, positive = positive)

    def add_constraint(self, *args, **kwargs):
        if isinstance(args[0], Function):
            if isinstance(args[0],InEquality):
                assert args[0].op in [Relations.ge, Relations.le], \
                    "Only '>=' and '<=' arre allowed!"
            c = args[0]
        else:
            c = Relation(*args,**kwargs)
        self.constraints.append(c)

    @property
    def variables(self):
        return self.vmanager.target()

    def sync_variables(self):
        s = set()
        s.update(self.obj.variables)
        for c in self.constraints:
            s.update(c.variables)
        self.vmanager.add_variables(s)

    def shift_variables(self):
        vmap = dict()
        for v in self.vmanager.source():
            if not v.is_positive:
                s = str(v)
                expr = sum(sy.symbols([s + '__+', s + '__-'], positive = True))
                vmap[v] = expr
        self.vmanager.substitute(vmap)

    def get_system_variables(self):
        s = set()
        s.update(self.obj.variables)
        for c in self.constraints:
            s.update(c.variables)
        return list(s)

    def get_slack_variables(self, template = 'y_{}'):
        inequalities = list(filter(lambda c : isinstance(c,InEquality),
                            self.constraints))
        n = len(inequalities)
        strlist = [template.format(i) for i in range(n)]
        y = sy.symbols(strlist, positive = True)
        for i in range(n):
            inequalities[i].slack = y[i]
        return y

    def has_standardform(self):
        all_eq = all([isinstance(c, Equality) for c in self.constraints])
        all_pos = all([v.is_positive for v in self.get_system_variables()])
        return all_pos and all_eq

    def as_standard(self, inplace = False):
        #handle variables
        self.sync_variables()
        self.shift_variables()
        self.vmanager.add_variables(self.get_slack_variables())
        v = self.vmanager.source()
        n = len(v)
        x = list(sy.symbols(' '.join(['x_{}'.format(i) for i in range(n)]),
                            positive = True))
        c = list(sy.symbols(' '.join(['c_{}'.format(i) for i in range(n)])))
        x_v = {x_ : v_ for x_, v_ in zip(x, v)}
        self.vmanager.substitute(vmap = x_v, inverse = True)
        v.append(1)
        x.append(1)
        c.append(sy.symbols('c'))
        x_c = {x_ : c_ for x_, c_ in zip(x,c)}
        template = np.inner(c,x)
        x.pop(-1)
        v.pop(-1)
        vmap = self.vmanager.vmap

        def redefine(fnc):
            expr = fnc.expr.subs([(v, expr) for v, expr in vmap.items()])
            fnc_coeffs = coefficients(expr = expr, normalize=True)
            coeffs = defaultdict(lambda : 0)
            coeffs.update({x_c[x] : c for x, c in fnc_coeffs.items()})
            if isinstance(fnc, InEquality):
                coeffs[x_c[fnc.slack]] = 1 if fnc.op == Relations.ge else -1
            expr = template.subs([(c_, coeffs[c_]) for c_ in c])
            if isinstance(fnc, Equality):
                return Equality(expr, variables = x, vmap = x_v)
            elif isinstance(fnc, InEquality):
                eq = Equality(expr, variables = x, vmap = x_v)
                eq.slack = fnc.slack
                return eq
            elif isinstance(fnc, Function):
                return Function(expr, variables = x, vmap = x_v)
            else:
                return None

        obj = redefine(self.obj)
        constraints = [redefine(c) for c in self.constraints]
        return LLP(obj = obj, constraints=constraints, variables = x)

    def eval_constraints(self,x):
        return np.array([c.f0(x) for c in self.constraints],
                        dtype = np.float32)

    def feasible(self, x = None) -> bool:
        c = [c.relate(x) for c in self.constraints]
        return all(c) and all(x >= 0)

    @staticmethod
    def basic_solution(A = None, b = None, order = None, **kwargs):
        m, n = A.shape
        r = n - m
        assert r > 0

        stop = False
        try:
            if order is not None:
                if isinstance(order, Iterable):
                    permutations = iter([order])
            else:
                order = [i for i in range(n)]
                permutations = multiset_permutations(order)
            while not stop:
                order = next(permutations)
                A_ = A[:,order]
                B_ = A_[:,:m]
                try:
                    B_inv = np.linalg.inv(B_)
                    xB = np.matmul(B_inv,b)
                    stop = all(xB >= 0)
                except LinAlgError:
                    """
                    If there is no error, it means that calculation
                    of xB was succesful, which is only possible if the
                    current permutation defines a positive definite submatrix.
                    Note that this is cheaper than checking the eigenvalues,
                    since it only requires the eigenvalues to be all positive,
                    and does not involve calculating their actual values.
                    """
                    pass
        except StopIteration:
            """
            There is no permutation of columns that would produce a regular
            mxm submatrix
                -> there is no feasible basic solution
                    -> there is no feasible solution
            """
            pass
        finally:
            if stop:
                N_ = A_[:,m:]
                xN = np.zeros(r, dtype = np.float32)
                return B_, B_inv, N_, xB, xN, order
            else:
                return None

    @staticmethod
    def solve_standard_form(A, b, c, order = None, tol = 1e-10):
        m, n = A.shape
        r = n - m
        assert r > 0
        basic = LLP.basic_solution(A, b, order = order)
        if basic:
            B, B_inv, N, xB, xN, order = basic
            c_ = c[order]
            cB = c_[:m]
            cN = c_[m:]
        else:
            return None

        def current_result(message : str = None, fail = False):
            if message:
                print(message)
            if not fail:
                return np.concatenate((xB, xN))[np.argsort(order)]
            return None

        def unit_basis_vector(length, index = 0, value = 1.0):
            return value * np.bincount([index], None, length)

        degenerate = False
        while True:
            if degenerate:
                #The objective could be decreased, but only on the expense
                #of violating positivity of the standard variables.
                #Hence, the solution is degenerate.
                return current_result('degenerate solution')

            #calculate reduced costs
            W = np.matmul(B_inv,N)
            reduced_costs = cN - np.matmul(cB,W)
            nEntering = np.count_nonzero(reduced_costs < 0)
            if nEntering == 0:
                #The objective can not be further reduced.
                #There was only one basic solution, which is
                #a unique optimizer.
                d = np.count_nonzero(reduced_costs > 0)
                if d < len(reduced_costs):
                    return current_result('multiple solutions')
                else:
                    return current_result('unique solution')
            #If we reach this line, reduction of the objective is possible,
            #although maybe indefinitely. If the objective can be decreased,
            #but only on the expense of violating feasibility, the
            #solution is degenerate.
            degenerate = True

            #Candidates for entering index are the indices of the negative
            #components of the vector of reduced costs.
            i_entering = np.argsort(reduced_costs)[:nEntering]
            for i_enter in i_entering:
                #basis vector for the current entering variable
                b_enter = unit_basis_vector(r, i_enter, 1.0)

                #w = vector of decrements of the current solution xB
                #Only positive values are a threat to feasibility, and we
                #need to tell which of the components of xB vanishes first,
                #which, since all components of xB are posotive,
                #has to do with the positive components only.
                w_enter = np.matmul(W,b_enter)
                i_leaving = np.argwhere(w_enter > 0)
                if len(i_leaving) == 0:
                    #step size could be indefinitely increased in this
                    #direction without violating feasibility, there is
                    #no solution to the problem
                    return current_result('no solution', fail = True)
                vanishing_ratios = xB[i_leaving]/w_enter[i_leaving]
                #the variable that vanishes first is the one with the smallest
                #vanishing ratio
                i_leave = i_leaving.flatten()[np.argmin(vanishing_ratios)]

                #step size in the direction of current basis vector
                t = xB[i_leave]/w_enter[i_leave]

                #update_solution
                if abs(t) <= tol:
                    #Smallest vanishing ratio is zero, any step would
                    #result in an infeasible situation.
                    # -> go for the next entering variable
                    continue
                xB -= t*w_enter
                xN = t*b_enter

                #reorder
                order[m + i_enter], order[i_leave] = \
                    order[i_leave], order[m + i_enter]
                B[:, i_leave], N[:, i_enter] = \
                    N[:, i_enter], copy(B[:, i_leave])
                B_inv = np.linalg.inv(B)
                cB[i_leave], cN[i_enter] = cN[i_enter], cB[i_leave]
                xB[i_leave], xN[i_enter] = xN[i_enter], xB[i_leave]

                #break loop at the first meaningful (t != 0) decrease and
                #force recalculation of the vector of reduced costs
                degenerate = False
                break

    def solve(self, order = None, as_dict = False):
        P = self.as_standard()

        #calculate A,b and c
        x = P.variables
        n = len(x)
        zeros = np.zeros((n,), dtype = np.float32)
        b = - P.eval_constraints(zeros)
        A = []
        for c in P.constraints:
            coeffs = c.linear_coefficients(normalize = True)
            A.append(np.array([coeffs[x_] for x_ in x], dtype = np.float32))
        A = np.vstack(A)
        coeffs = P.obj.linear_coefficients(normalize = True)
        c = np.array([coeffs[x_] for x_ in x], dtype = np.float32)

        #calculate solution
        x_ = LLP.solve_standard_form(A, b, c, order = order)
        if x_ is not None:
            self.vmanager.substitute({v : val
                                     for v, val in zip(P.variables, x_)})
            if as_dict:
                return self.vmanager.vmap
            else:
                return np.array(list(self.vmanager.vmap.values()),
                                dtype = np.float32)
        else:
            return None


if __name__ == '__main__':

    """
    Example for unique solution
    (0, 6, 0, 4) --> 10
    The following order automatically creates
    a feasble solution : [0,2,3,1]
    """
    variables = ['x1', 'x2', 'x3', 'x4']
    x1, x2, x3, x4 = syms = sy.symbols(variables, positive = True)
    obj1 = Function(3*x1 + 9*x3 + x2 + x4, variables = syms)
    eq11 = Equality(x1 + 2*x3 + x4 - 4, variables = syms)
    eq12 = Equality(x2 + x3 - x4 - 2, variables = syms)
    P1 = LLP(cost = obj1, constraints = [eq11, eq12], variables = syms)
    x_1 = P1.solve(order = [0,2,3,1])

    """
    Example for degenerate solution.
    (0, 2, 0, 0)
    """
    variables = ['x1', 'x2', 'x3', 'x4']
    x1, x2, x3, x4 = syms = sy.symbols(variables, positive = True)
    obj2 = Function(3*x1 + x2 + 9*x3 + x4, variables = syms)
    eq21 = Equality(x1 + 2*x3 + x4, variables = syms)
    eq22 = Equality(x2 + x3 - x4 - 2, variables = syms)
    P2 = LLP(cost = obj2, constraints = [eq21, eq22], variables = syms)
    x_2 = P2.solve()

    """
    Example for no solution.
    """
    variables = ['x1', 'x2', 'x3', 'x4']
    x1, x2, x3, x4 = syms = sy.symbols(variables, positive = True)
    obj3 = Function(-3*x1 + x2 + 9*x3 + x4, variables = syms)
    eq31 = Equality(x1 - 2*x3 - x4 + 2, variables = syms)
    eq32 = Equality(x2 + x3 - x4 - 2, variables = syms)
    P3 = LLP(cost = obj3, constraints = [eq31, eq32], variables = syms)
    x_3 = P3.solve()

    """
    Example for multiple solutions.
    (0, 1, 1, 0), (0, 4, 0, 2)
    """
    variables = ['x1', 'x2', 'x3', 'x4']
    x1, x2, x3, x4 = syms = sy.symbols(variables, positive = True)
    obj4 = Function(3*x1 + 2*x2 + 8*x3 + x4, variables = syms)
    eq41 = Equality(x1 - 2*x3 - x4 + 2, variables = syms)
    eq42 = Equality(x2 + x3 - x4 - 2, variables = syms)
    P4 = LLP(cost = obj4, constraints = [eq41, eq42], variables = syms)
    x_4 = P4.solve()
