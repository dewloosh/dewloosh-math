# -*- coding: utf-8 -*-
from typing import Iterable, Union
from decimal import Decimal
from copy import deepcopy
import numpy as np
from functools import reduce
from itertools import product
import operator
from dewloosh.core.abc import ABC_Safe

__all__ = ['Scalar', 'List', 'Array']


Scalar = Union[float, complex, Decimal]


class List(list):
    """
    This is a pure pythonic implementation of an Array. It is called a List
    to distinguish it from the object Array, which is built on numpy.ndarray.
    """

    def __init__(self, iterable: Iterable, *args, **kwargs):
        super().__init__(iterable)
        self._dtype = kwargs.pop('dtype', float)
        if 'shallow' not in args:
            self = List._unpack(self)
            self.convert()

    @staticmethod
    def from_base(comps, base: Iterable):
        pass

    @staticmethod
    def _unpack(iterable: Iterable, dtype=None):
        if dtype is None:
            try:
                dtype = iterable._dtype
            except Exception:
                dtype = float

        if not isinstance(iterable, List):
            iterable = List(iterable, 'shallow', dtype=dtype)

        for i, item in enumerate(iterable):
            if isinstance(item, Iterable):
                iterable[i] = List._unpack(item, dtype=dtype)
            else:
                iterable[i] = dtype(item)
        return iterable

    @staticmethod
    def nested(iterable: Iterable):
        for item in iterable:
            if isinstance(item, Iterable):
                return True
        return False

    def convert(self, dtype=None):
        if dtype is None:
            dtype = self._dtype
        for array in self.arrays():
            for i, val in enumerate(array):
                array[i] = dtype(val)
        return

    def values(self, iterator: Iterable = None):
        """
        Yields scalar values in a List.
        """
        if iterator is None:
            iterator = self
        for item in iterator:
            if isinstance(item, Iterable):
                for subitem in self.values(item):
                    yield subitem
            else:
                yield item
        return

    def keys(self):
        """
        Yields addresses of values.
        """
        for address, _ in self.items():
            yield address

    def item(self, key: Iterable):
        command = 'self' + len(key)*'[{}]'
        return eval(command.format(*key))

    def setitem(self, key: Iterable, value):
        command = 'self' + len(key)*'[{}]' + '={}'
        exec(command.format(*key, self.dtype(value)))

    def _shape(self, iterator: Iterable = None, address: list = None):
        """
        Yields key (=address), value pairs.
        """
        if iterator is None:
            iterator = self
        if address is None:
            address = [0]
        else:
            address.append(0)

        yield {'level': len(address)-1, 'length': len(iterator)}
        for i, item in enumerate(iterator):
            address[-1] = i
            if isinstance(item, Iterable):
                for subitem in self._shape(item, deepcopy(address)):
                    yield subitem
        return

    @property
    def shape(self):
        size = self.size
        shape = []
        for _, value in size.items():
            if not len(set(value)) == 1:
                shape.append(value)
            else:
                shape.append(value[0])
        return tuple(shape)

    @property
    def size(self):
        size = {}
        for d in self._shape():
            level = d['level']
            length = d['length']
            if level in size:
                size[level].append(length)
            else:
                size[level] = [length]
        return size

    def items(self, iterator: Iterable = None, address: list = None):
        """
        Yields key (=address), value pairs.
        """
        if iterator is None:
            iterator = self
        if address is None:
            address = [0]
        else:
            address.append(0)

        for i, item in enumerate(iterator):
            address[-1] = i
            if isinstance(item, Iterable):
                for subitem in self.items(item, deepcopy(address)):
                    yield subitem
            else:
                yield tuple(address), item
        return

    def iterator(self, dim=1, *args, **kwargs):
        """
        Returns the innermost arrays.
        """
        iterable = kwargs.get("iterable", self)
        address = kwargs.get('address', [0])
        if 'address' in kwargs:
            address.append(0)
        depth = kwargs.get('depth', self.dim)
        assert depth >= dim
        level = len(address)-1

        if depth-level == dim:
            yield tuple(address[:-1]), iterable
        else:
            level += 1
            if depth-level == dim:
                for i, item in enumerate(iterable):
                    address[-1] = i
                    yield tuple(address), item
            else:
                for i, item in enumerate(iterable):
                    address[-1] = i
                    for subitem in self.iterator(dim,
                                                 iterable=item,
                                                 address=deepcopy(address),
                                                 depth=depth):
                        yield subitem
        return

    def arrays(self, dim=1, *args, **kwargs):
        for _, array in self.iterator(dim, *args, **kwargs):
            yield array

    def flatten(self):
        items, keys = [], []
        for key, item in self.items():
            items.append(item)
            keys.append(key)
        return keys, items

    @property
    def dim(self):
        """
        Returns the dimension of the List.
        """
        g = self.keys()
        d = len(next(g))
        while True:
            try:
                if not len(next(g)) == d:
                    raise TypeError('Elements must have the same basis.')
            except StopIteration:
                return d

    @property
    def dtype(self):
        return self._dtype

    @property
    def array_like(self):
        shp = self.shape
        return not List.nested(shp)

    def to_Array(self, order='C'):
        """
        Converts the List object to a Array, which is basically a numpy array,
        with some additional features.
        """
        res = np.zeros(self.shape)
        for address, value in self.items():
            try:
                res[address] = value
            except Exception:
                print(address)
        if order == 'C':
            return Array(np.ascontiguousarray(res), dtype=self._dtype)
        elif order == 'F':
            return Array(np.asfortranarray(res), dtype=self._dtype)
        else:
            return Array(np.asanyarray(res), dtype=self._dtype)

    @staticmethod
    def fill(shape: tuple, fillval=None):
        assert isinstance(shape, Iterable)
        lshape = list(shape)
        res = [fillval for i in range(lshape.pop(-1))]
        for d in reversed(lshape):
            res = [deepcopy(res) for i in range(d)]
        return List(res, dtype=type(fillval))

    @staticmethod
    def zeros(shape: tuple, dtype=float):
        return List.fill(shape, dtype(0))

    def indices(self, shape: tuple = None):
        if shape is None:
            shape = self.shape
        return List(np.indices(shape), dtype=int)

    @staticmethod
    def IDarray(shape: tuple, dtype=int):
        return List(np.arange(np.prod(shape)).reshape(*shape), dtype=dtype)

    @staticmethod
    def none(shape: tuple):
        return List.fill(shape, None)

    @staticmethod
    def ones(shape: tuple):
        return List.fill(shape, 1.0)

    def __add__(self, other: 'List') -> 'List':
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        res = deepcopy(self)
        for key, array in other.iterator():
            a = res.item(key)
            for i, val in enumerate(array):
                a[i] += val
        return res

    def __sub__(self, other: 'List') -> 'List':
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        res = deepcopy(self)
        for key, array in other.iterator():
            a = res.item(key)
            for i, val in enumerate(array):
                a[i] -= val
        return res

    def __mul__(self, other):
        if isinstance(other, Iterable):
            pass
        else:
            res = deepcopy(self)
            for _, array in res.iterator():
                for i, val in enumerate(array):
                    array[i] = val*other
            return res

    def __rmul__(self, other):
        if isinstance(other, Iterable):
            pass
        else:
            return self.__mul__(other)

    def __eq__(self, other):
        if not self.shape == other.shape:
            return False
        assert self.dtype == other.dtype
        for i, j in zip(self.values(), other.values()):
            if i != j:
                return False
        return True

    @staticmethod
    def einsum(command, Arrays, *args, **kwargs):
        """
        Implies the Einstein summa convention.

        a = List([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
        v = List.einsum('ijj',[a])
        s = List.einsum('i,i',[v,v])
        """
        # separation of commands
        if isinstance(command, str):
            exprs = command.split(kwargs.pop('separator', ','))
        else:
            exprs = command
        nTerm = len(exprs)
        assert len(Arrays) == nTerm

        # positions of indices
        indices = dict()
        addresses = []
        for Array, expr in enumerate(exprs):
            addresses.append([0 for i in expr])
            for index, character in enumerate(expr):
                if character in indices:
                    indices[character].append((Array, index))
                else:
                    indices[character] = [(Array, index)]

        # order of indices
        order = deepcopy(indices)
        if 'abc' not in kwargs:
            abc = np.sort(list(order.keys()))
        else:
            abc = np.array(list(kwargs['abc']))
        for key in order.keys():
            order[key] = int(np.where(abc == key)[0])

        # ranges
        ranges = [None for i in abc]
        for expr, Array in zip(exprs, Arrays):
            for pos, index in enumerate(expr):
                if ranges[order[index]] is None:
                    ranges[order[index]] = Array.shape[pos]

        # repeated
        count = np.zeros(abc.shape, dtype=np.int16)
        for expr in exprs:
            for index in expr:
                count[order[index]] += 1
        dummy = abc[np.where(count > 1)]
        free = abc[np.where(count == 1)]

        # result
        if len(free) > 0:
            newshape = tuple([ranges[order[i]] for i in free])
            res = List.zeros(newshape, Arrays[0].dtype)
            target = product(*[range(ranges[order[f]]) for f in free])
        else:
            res = List.zeros((1,), Arrays[0].dtype)
            target = [(0,)]

        for address_f in target:
            for i, f in enumerate(free):
                for term, pos in indices[f]:
                    addresses[term][pos] = address_f[i]

            for address_d in product(*[range(ranges[order[d]])
                                       for d in dummy]):
                for j, d in enumerate(dummy):
                    for term, pos in indices[d]:
                        addresses[term][pos] = address_d[j]

                values = [t.item(a) for t, a in zip(Arrays, addresses)]
                val = reduce(operator.mul, values)
                res.setitem(address_f, val)

        return res

    def __call__(self, cmd: str):
        return List.einsum(cmd, [self])


class Array(ABC_Safe, np.ndarray):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __new__(cls, v: Iterable, *args, **kwargs):
        if not isinstance(v, np.ndarray):
            v = List(v, 'shallow', *args, **kwargs).to_Array()
        kwargs['buffer'] = v
        kwargs['shape'] = v.shape
        kwargs['dtype'] = v.dtype
        return super().__new__(cls, *args, **kwargs)

    def arrays(self, iterable: Iterable=None, address: list=None,
               cls=None):
        """
        Returns the innermost arrays.
        """
        if iterable is None:
            iterable = self
        if address is None:
            address = [0]
        else:
            address.append(0)
        if cls is None:
            cls = type(self)

        for i, item in enumerate(iterable):
            address[-1] = i
            if isinstance(item, Iterable):
                for subitem in self.arrays(item, deepcopy(address), cls):
                    yield subitem
            else:
                yield tuple(address[:-1]), cls(iterable)
                break
        return

    @property
    def dim(self):
        return self.ndim

    @property
    def depth(self):
        return self.dim

    @property
    def inv(self):
        return Array(np.linalg.inv(self))

    @staticmethod
    def zeros(shape: tuple, *args, **kwargs):
        return Array(np.zeros(shape), *args, **kwargs)

    @staticmethod
    def einsum(subscripts, *operands, **kwargs):
        return Array(np.einsum(subscripts, *operands, **kwargs))

    @staticmethod
    def stack(arrays, axis=0):
        return Array(np.stack(arrays, axis))

    def matmul(self, v):
        return Array(np.matmul(self, v))

    @staticmethod
    def IDarray(shape: tuple):
        return Array(np.arange(np.prod(shape)).reshape(*shape))

    @staticmethod
    def indices(shape: tuple):
        return Array(np.indices(shape))


if __name__ == '__main__':
    # %%

    a = List([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    print('\n---as List')
    print('dimension : {}'.format(a.dim))
    print('shape : {}'.format(a.shape))
    a_ = List.einsum('ijk', [a])
    v = List.einsum('ijj', [a_])
    s = List.einsum('i,i', [v, v])

    print('--> values')
    for val in a.values():
        print(val)

    print('--> keys')
    for k in a.keys():
        i, j, k = k
        print(k)

    print('--> items')
    for i in a.items():
        print(i)

    print('--> arrays')
    for arr in a.arrays():
        print(arr)

    b = List.fill(a.shape, 5.0)
    c = a + b
    d = c - b
    e = d - a
    aa = a*2

    t = Array(a)
    print('\n---as Array')
    print('dimension : {}'.format(t.dim))
    print('shape : {}'.format(t.shape))

    print('--> arrays')
    for arr in t.arrays():
        print(arr)

    # %%
    size = (1, 1, 1)
    dsize = (0.5, 0.5, 0.5)
    dim = len(size)
    assert len(dsize) == dim
    num = [int(np.ceil(size[i]/dsize[i])) for i in range(dim)]
    numP = [n+1 for n in num]
    # ---

    def expand():
        avg = np.average(size)
        size.append(avg)
        dsize.append(avg)
        num.append(1)
        numP.append(2)

    if dim < 2:
        expand()
    if dim < 3:
        expand()

    # indices = Array(np.ascontiguousarray(np.einsum('ijkl -> jkli',np.indices(numP))))
    # xyz = np.array([np.linspace(0.,size[i],num = n) for i,n in enumerate(num)])
    pointindices = Array.indices(numP)
    coords = np.array([pointindices[i]*dsize[i] for i in range(dim)])

    pointIDs = Array.IDarray(numP)
    coords = Array(np.ascontiguousarray(np.einsum('ijkl -> jkli', coords)))
    for a, pos in coords.arrays():
        pID = pointIDs[a]
        print('address : {}, ID : {}, pos : {}'.format(a, pID, pos))

    IDmap = [0, 4, 6, 2, 1, 5, 7, 3]
    cellIDs = List.IDarray(num)

    def cube(a): return List(
        pointIDs[a[0]: a[0]+2, a[1]: a[1]+2, a[2]: a[2]+2], dtype=int)
    for a, v in cellIDs.items():
        print('a : {}, v : {}, IDs : {}'.format(a, v, cube(a)))

    cube0 = cube((0, 0, 0))
    _, cube0f = cube0.flatten()
    cubeID = List.IDarray((2, 2, 2))
    cubeID.shape

    for a, item in cubeID.iterator(dim=0):
        print('{} : {}'.format(a, item))

    cubeind = cubeID.indices()

    # %%
    # t = Array([[1,4,5],[4,2,6],[5,6,3]])

    # #%%
    # #vector in original base
    # v = Array([7,2])

    # #target base
    # e1 = Array([1,3])
    # e2 = Array([4,0])

    # #covariant componets
    # T_co = Array.stack([e1,e2])
    # v_co = T_co.matmul(v)

    # #covariant components = compoments of v in new base
    # T_contra = T_co.inv
    # v_contra = v.matmul(T_contra)

    # #dual base
    # e1_d = T_contra[0,:]
    # e2_d = T_contra[1,:]

    # %%
    # v = np.array([7,2])
    # e1 = np.array([1,0])
    # e2 = np.array([0,1])

    # e1_ = np.array([1,3])
    # e2_ = np.array([4,0])

    # v_col = np.matmul(np.matmul(np.linalg.inv(np.stack([e1_,e2_],axis = -1)),
    #                np.stack([e1,e2],axis = -1)),v)

    # v_row = np.matmul(v,np.matmul(np.stack([e1,e2]),np.linalg.inv(np.stack([e1_,e2_]))))

    # v_ein = np.einsum('j,ji,ji',v,np.stack([e1,e2]),np.linalg.inv(np.stack([e1_,e2_])))
