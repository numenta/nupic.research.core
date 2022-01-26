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


class SparseBinaryMatrix(_nupic.SparseBinaryMatrix):
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], str):
                super().__init__(1)
                self.fromCSR(args[0])
            elif isinstance(args[0], np.ndarray) or hasattr(args[0], '__iter__'):
                super().__init__(1)
                self.fromDense(np.asarray(args[0]))
            elif isinstance(args[0], int):
                super().__init__(args[0])
            elif isinstance(args[0], _SM_01_32_32):
                super().__init__(1)
                self.copy(args[0])
            elif isinstance(args[0], _SparseMatrix32):
                super().__init__(1)
                nz_i,nz_j,nz_v = args[0].getAllNonZeros(True)
                self.setAllNonZeros(args[0].nRows(), args[0].nCols(), nz_i, nz_j)
        elif len(args) == 2:
            if isinstance(args[0], int) and isinstance(args[1], int):
                super().__init__(args[0], args[1])

    def __str__(self):
        return str(self.toDense())

    def __setstate__(self, inString):
        self.fromCSR(inString)

    def __getitem__(self, index):
        return np.float32(self.get(index[0], index[1]))

    def __setitem__(self, index, value):
        self.set(index[0], index[1], value)

    def fromDense(self, m):
        m = np.asarray(m, dtype="bool")
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

    def write(self, pyBuilder):
        """Serialize the SparseMatrix instance using capnp.

        :param: Destination SparseMatrixProto message builder
        """
        # NOTE need to import capnp first to activate the magic necessary for
        # SparseMatrixProto
        import capnp
        from nupic.proto.SparseBinaryMatrixProto_capnp import SparseBinaryMatrixProto
        # Capnp reader traveral limit (see capnp::ReaderOptions)
        _TRAVERSAL_LIMIT_IN_WORDS = 1 << 63

        reader = SparseBinaryMatrixProto.from_bytes(
            self._writeAsCapnpPyBytes(),
            traversal_limit_in_words=_TRAVERSAL_LIMIT_IN_WORDS)
        pyBuilder.from_dict(reader.to_dict())  # copy

    @classmethod
    def getSchema(cls):
        """ Get Cap'n Proto schema.
        :return: Cap'n Proto schema
        """
        return SparseBinaryMatrixProto

    def read(self, proto):
        """Initialize the SparseMatrix instance from the given SparseMatrixProto
        reader.

        :param proto: SparseMatrixProto message reader containing data from a previously
                      serialized SparseMatrix instance.
        """
        self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2
