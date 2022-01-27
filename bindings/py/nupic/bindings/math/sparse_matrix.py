# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np

import _nupic

from .sparse_binary_matrix import SparseBinaryMatrix


class SparseMatrix(_nupic.SparseMatrix):
    def __init__(self, *args):
        """
        Constructs a new SparseMatrix from the following available arguments:
                      SparseMatrix(): An empty sparse matrix with 0 rows and columns.
          SparseMatrix(nrows, ncols): A zero sparse matrix with the
                                      specified rows and columns.
          SparseMatrix(SparseMatrix): Copies an existing sparse matrix.
                SparseMatrix(string): Loads a SparseMatrix from its serialized form.
           SparseMatrix(numpy.array): Loads a SparseMatrix from a numpy array.
           SparseMatrix([[...],[...]]): Creates an array from a list of lists.
        """
        serialized,dense,from01,fromstr3f = None,None,False,False
        fromSpecRowCols = False

        if (len(args) == 3) and isinstance(args[0], SparseMatrix):
            fromSpecRowCols = True

        if (len(args) == 1):
            if isinstance(args[0], str):
                serialized = args[0]
                args = tuple()
            elif isinstance(args[0], np.ndarray):
                dense = args[0]
                args = tuple()
            elif hasattr(args[0], '__iter__'):
                dense = args[0]
                args = tuple()
            elif isinstance(args[0], SparseBinaryMatrix):
                from01 = True

        if from01 or fromSpecRowCols:
            this = super().__init__(1, 1)
        else:
            this = super().__init__(*args)

        try:
            self.this.append(this)
        except:
            self.this = this

        if serialized is not None:
            s = serialized.split(None, 1)
            self.fromPyString(serialized)

        elif dense is not None:
            self.fromDense(np.asarray(dense,dtype=np.float32))

        elif from01:
            nz_i,nz_j = args[0].getAllNonZeros(True)
            nz_ones = np.ones((len(nz_i)))
            self.setAllNonZeros(args[0].nRows(), args[0].nCols(), nz_i, nz_j, nz_ones)

        elif fromstr3f:
            nz_i,nz_j,nz_v = args[1].getAllNonZeros(args[0], True)
            self.setAllNonZeros(args[1].nRows(), args[1].nCols(), nz_i,nz_j,nz_v)

        elif fromSpecRowCols:
            if args[2] == 0:
                self.__initializeWithRows(args[0], args[1])
            elif args[2] == 1:
                self.__initializeWithCols(args[0], args[1])

    def __getitem__(self, index):
        return np.float32(self.get(index[0], index[1]))

    def __setitem__(self, index, value):
        self.set(index[0], index[1], value)

    def __str__(self):
        return str(self.toDense())

    def _setShape(self, *args):
        if len(args) == 1:
            self.resize(*(args[0]))
        elif len(args) == 2:
            self.resize(*args)
        else:
            raise RuntimeError("Error: setShape(rows, cols) or setShape((rows, cols))")
        shape = property(fget=lambda self: (self.nRows(), self.nCols()), fset=_setShape,
                         doc="rows, cols")

    def getTransposed(self):
        result = self.__class__()
        self.transpose(result)
        return result

    def __neg__(self):
        result = SparseMatrix(self)
        result.negate()
        return result

    def __abs__(self):
        result = SparseMatrix(self)
        result.abs()
        return result

    def __iadd__(self, other):
        t = type(other).__name__
        if t == "float32":
            self.__add(other)
        elif t == 'ndarray':
            self.add(SparseMatrix(other))
        elif t == 'SparseMatrix':
            self.add(other)
        else:
            raise Exception("Can't use type: " + t)
        return self

    def __add__(self, other):
        arg = None
        result = SparseMatrix(self)
        t = type(other).__name__
        if t == "float32":
            result.__add(other)
        elif t == 'ndarray':
            result.add(SparseMatrix(other))
        elif t == 'SparseMatrix':
            result.add(other)
        else:
            raise Exception("Can't use type: " + t)
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        t = type(other).__name__
        if t == "float32":
            self.__subtract(other)
        elif t == 'ndarray':
            self.subtract(SparseMatrix(other))
        elif t == 'SparseMatrix':
            self.subtract(other)
        else:
            raise Exception("Can't use type: " + t)
        return self

    def __sub__(self, other):
        result = SparseMatrix(self)
        t = type(other).__name__
        if t == "float32":
            result.__subtract(other)
        elif t == 'ndarray':
            result.subtract(SparseMatrix(other))
        elif t == 'SparseMatrix':
            result.subtract(other)
        else:
            raise Exception("Can't use type: " + t)
        return result

    def __rsub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        t = type(other).__name__
        if t == "float32":
            self.__multiply(other)
        elif t == 'SparseMatrix':
            self.multiply(other)
        else:
            raise Exception("Can't use type: " + t)
        return self

    def __mul__(self, other):
        t = type(other).__name__
        arg = other
        result = None
        if t == "float32":
            result = SparseMatrix(self)
            result.__multiply(arg)
        elif t == 'ndarray':
            if arg.ndim == 1:
                result = np.array(self.rightVecProd(arg))
            elif arg.ndim == 2:
                arg = SparseMatrix(other)
                result = SparseMatrix()
                self.multiply(arg, result)
            else:
                raise Exception("Wrong ndim: " + str(arg.ndim))
        elif t == 'SparseMatrix':
            if other.nCols() == 1:
                if self.nRows() == 1:
                    result = self.rightVecProd(other.getCol(0))[0]
                else:
                    result_list = self.rightVecProd(other.getCol(0))
                    result = SparseMatrix(self.nRows(), 0)
                    result.addCol(result_list)
            else:
                result = SparseMatrix()
                self.multiply(arg, result)
        else:
            raise Exception("Can't use type: " + t + " for multiplication")
        return result

    def __rmul__(self, other):
        t = type(other).__name__
        arg = other
        result = None
        if t == "float32":
            result = SparseMatrix(self)
            result.__multiply(arg)
        elif t == 'ndarray':
            if arg.ndim == 1:
                result = np.array(self.leftVecProd(arg))
            elif arg.ndim == 2:
                arg = SparseMatrix(other)
                result = SparseMatrix()
                arg.multiply(self, result)
            else:
                raise Exception("Wrong ndim: " + str(arg.ndim))
        elif t == 'SparseMatrix':
            if other.nRows() == 1:
                if self.nCols() == 1:
                    result = self.leftVecProd(other.getRow(0))[0]
                else:
                    result_list = self.leftVecProd(other.getRow(0))
                    result = SparseMatrix(self.nCols(), 0)
                    result.addRow(result_list)
            else:
                result = SparseMatrix()
                arg.multiply(self, result)
        else:
            raise Exception("Can't use type: " + t + " for multiplication")
        return result

    def __idiv__(self, other):
        t = type(other).__name__
        if t == "float32":
            self.__divide(other)
        else:
            raise Exception("Can't use type: " + t)
        return self

    def __div__(self, other):
        t = type(other).__name__
        if t == "float32":
            result = SparseMatrix(self)
            result.__divide(other)
            return result
        else:
            raise Exception("Can't use type: " + t)

    def fromDense(self, m):
        m = np.asarray(m, dtype="float32")
        self._fromDense(m.shape[0], m.shape[1], m)

    def rightVecSumAtNZ(self, denseArray, out=None):
        denseArray = np.asarray(denseArray, dtype="float32")

        if out is None:
            out = np.empty(self.nRows(), dtype="float32")
        else:
            assert out.dtype == "float32"

        self._rightVecSumAtNZ(denseArray, out)

        return out

    def rightVecSumAtNZ_fast(self, denseArray, out):
        """
        Deprecated. Use rightVecSumAtNZ with an 'out' specified.
        """
        self.rightVecSumAtNZ(denseArray, out)

    def rightVecSumAtNZSparse(self, sparseBinaryArray, out=None):
        sparseBinaryArray = np.asarray(sparseBinaryArray, dtype="uint32")

        if out is None:
            out = np.empty(self.nRows(), dtype="int32")
        else:
            assert out.dtype == "int32"

        self._rightVecSumAtNZSparse(sparseBinaryArray, out)

        return out

    def rightVecSumAtNZGtThreshold(self, denseArray, threshold, out=None):
        denseArray = np.asarray(denseArray, dtype="float32")

        if out is None:
            out = np.empty(self.nRows(), dtype="float32")
        else:
            assert out.dtype == "float32"

        self._rightVecSumAtNZGtThreshold(denseArray, threshold, out)

        return out

    def rightVecSumAtNZGtThreshold_fast(self, denseArray, threshold, out):
        """
        Deprecated. Use rightVecSumAtNZGtThreshold with an 'out' specified.
        """
        self.rightVecSumAtNZGtThreshold(denseArray, threshold, out)

    def rightVecSumAtNZGtThresholdSparse(self, sparseBinaryArray, threshold, out=None):
        sparseBinaryArray = np.asarray(sparseBinaryArray, dtype="uint32")

        if out is None:
            out = np.empty(self.nRows(), dtype="int32")
        else:
            assert out.dtype == "int32"

        self._rightVecSumAtNZGtThresholdSparse(sparseBinaryArray, threshold, out)

        return out

    def rightVecSumAtNZGteThreshold(self, denseArray, threshold, out=None):
        denseArray = np.asarray(denseArray, dtype="float32")

        if out is None:
            out = np.empty(self.nRows(), dtype="float32")
        else:
            assert out.dtype == "float32"

        self._rightVecSumAtNZGteThreshold(denseArray, threshold, out)

        return out

    def rightVecSumAtNZGteThresholdSparse(self, sparseBinaryArray, threshold, out=None):
        sparseBinaryArray = np.asarray(sparseBinaryArray, dtype="uint32")

        if out is None:
            out = np.empty(self.nRows(), dtype="int32")
        else:
            assert out.dtype == "int32"

        self._rightVecSumAtNZGteThresholdSparse(sparseBinaryArray, threshold, out)

        return out

    def incrementNonZerosOnOuter(self, rows, cols, delta):
      self._incrementNonZerosOnOuter(np.asarray(rows, dtype="uint32"),
                                     np.asarray(cols, dtype="uint32"),
                                     delta)

    def incrementNonZerosOnRowsExcludingCols(self, rows, cols, delta):
      self._incrementNonZerosOnRowsExcludingCols(np.asarray(rows, dtype="uint32"),
                                                 np.asarray(cols, dtype="uint32"),
                                                 delta)

    def setZerosOnOuter(self, rows, cols, value):
      self._setZerosOnOuter(np.asarray(rows, dtype="uint32"),
                            np.asarray(cols, dtype="uint32"),
                            value)

    def setRandomZerosOnOuter(self, rows, cols, numNewNonZeros, value, rng):
      if isinstance(numNewNonZeros, numbers.Number):
        self._setRandomZerosOnOuter_singleCount(
          np.asarray(rows, dtype="uint32"),
          np.asarray(cols, dtype="uint32"),
          numNewNonZeros,
          value,
          rng)
      else:
        self._setRandomZerosOnOuter_multipleCounts(
          np.asarray(rows, dtype="uint32"),
          np.asarray(cols, dtype="uint32"),
          np.asarray(numNewNonZeros, dtype="int32"),
          value,
          rng)

    def increaseRowNonZeroCountsOnOuterTo(self, rows, cols, numDesiredNonZeros,
                                          initialValue, rng):
      self._increaseRowNonZeroCountsOnOuterTo(
        np.asarray(rows, dtype="uint32"),
        np.asarray(cols, dtype="uint32"),
        numDesiredNonZeros, initialValue, rng)

    def clipRowsBelowAndAbove(self, rows, a, b):
      self._clipRowsBelowAndAbove(np.asarray(rows, dtype="uint32"),
                                  a,
                                  b)

    def nNonZerosPerRow(self, rows=None):
      if rows is None:
        return self._nNonZerosPerRow_allRows()
      else:
        return self._nNonZerosPerRow(np.asarray(rows, dtype="uint32"))

    def nNonZerosPerRowOnCols(self, rows, cols):
      rows = np.asarray(rows, dtype="uint32")
      cols = np.asarray(cols, dtype="uint32")
      return self._nNonZerosPerRowOnCols(rows, cols)

    def write(self, pyBuilder):
        """Serialize the SparseMatrix instance using capnp.

        :param: Destination SparseMatrixProto message builder
        """
        import capnp
        from nupic.proto.SparseMatrixProto_capnp import SparseMatrixProto
        # Capnp reader traveral limit (see capnp::ReaderOptions)
        _TRAVERSAL_LIMIT_IN_WORDS = 1 << 63

        reader = SparseMatrixProto.from_bytes(
            self._writeAsCapnpPyBytes(),
            traversal_limit_in_words=_TRAVERSAL_LIMIT_IN_WORDS)
        pyBuilder.from_dict(reader.to_dict())  # copy

    @classmethod
    def getSchema(cls):
        """ Get Cap'n Proto schema.
        :return: Cap'n Proto schema
        """
        return SparseMatrixProto

    def read(self, proto):
        """Initialize the SparseMatrix instance from the given SparseMatrixProto
        reader.

        :param proto: SparseMatrixProto message reader containing data from a previously
                      serialized SparseMatrix instance.
        """
        self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2
