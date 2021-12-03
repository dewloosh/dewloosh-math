# -*- coding: utf-8 -*-
from polydata.math.function.function import Function
from enum import Enum
from polydata.tools import getasany
import operator as op
from typing import TypeVar, Callable

__all__ = ['Equality', 'InEquality']


class Relations(Enum):
    eq = '='
    gt = '>'
    ge = '>='
    lt = '<'
    le = '<='

    def to_op(self):
        return _rel_to_op[self]


_rel_to_op = {
    Relations.eq: op.eq,
    Relations.gt: op.gt,
    Relations.ge: op.ge,
    Relations.lt: op.lt,
    Relations.le: op.le
}

RelationType = TypeVar('RelationType', str, Relations, Callable)


class Relation(Function):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = None
        self.opfunc = None
        op = getasany(['op', 'operator'], None, **kwargs)
        if op:
            if isinstance(op, str):
                self.op = Relations(op)
            elif isinstance(op, Relations):
                self.op = op
            elif isinstance(op, Callable):
                self.opfunc = op
                self.op = None
        else:
            self.op = Relations.eq
        if op and isinstance(self.op, Relations):
            self.opfunc = self.op.to_op()
        self.slack = 0

    @property
    def operator(self):
        return self.op

    def to_eq(self):
        raise NotImplementedError

    def relate(self, *args, **kwargs):
        return self.opfunc(self.f0(*args, **kwargs), 0)

    def __call__(self, *args, **kwargs):
        return self.opfunc(self.f0(*args, **kwargs), 0)


class Equality(Relation):
    def __init__(self, *args, **kwargs):
        kwargs['op'] = Relations.eq
        super().__init__(*args, **kwargs)

    def to_eq(self):
        return self


class InEquality(Relation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_eq(self):
        raise


if __name__ == '__main__':

    gt = InEquality('x + y', op='>')
    print(gt([0.0, 0.0]))

    ge = InEquality('x + y', op='>=')
    print(ge([0.0, 0.0]))

    le = InEquality('x + y', op=lambda x, y: x <= y)
    print(le([0.0, 0.0]))

    lt = InEquality('x + y', op=lambda x, y: x < y)
    print(lt([0.0, 0.0]))
