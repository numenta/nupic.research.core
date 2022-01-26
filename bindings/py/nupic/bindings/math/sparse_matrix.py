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
