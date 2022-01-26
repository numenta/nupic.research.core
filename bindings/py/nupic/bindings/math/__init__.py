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

import numbers

import numpy as np

import _nupic


try:
    # NOTE need to import capnp first to activate the magic necessary for
    # RandomProto_capnp, etc.
    import capnp
except ImportError:
    capnp = None
else:
    from nupic.proto.RandomProto_capnp import RandomProto


# Capnp reader traveral limit (see capnp::ReaderOptions)
_TRAVERSAL_LIMIT_IN_WORDS = 1 << 63


class Random(_nupic.Random):
    def write(self, pyBuilder):
      """Serialize the Random instance using capnp.

      :param: Destination RandomProto message builder
      """
      reader = RandomProto.from_bytes(
          self._writeAsCapnpPyBytes(),
          traversal_limit_in_words=_TRAVERSAL_LIMIT_IN_WORDS)
      pyBuilder.from_dict(reader.to_dict())  # copy


    def read(self, proto):
      """Initialize the Random instance from the given RandomProto reader.

      :param proto: RandomProto message reader containing data from a previously
                    serialized Random instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2


def GetNTAReal():
    return np.float32


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
        reader = SparseMatrixProto.from_bytes(self._writeAsCapnpPyBytes(),
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


# Included for compatibility with code written for nupic.core
SM32 = SparseMatrix
SM_01_32_32 = SparseBinaryMatrix


class SparseMatrixConnections(_nupic.SparseMatrixConnections):
    def computeActivity(self, activeInputs, permanenceThreshold=None,
                        out=None):
        activeInputs = np.asarray(activeInputs, dtype="uint32")

        if out is None:
            out = np.empty(self.nSegments(), dtype="int32")
        else:
            assert out.dtype == "int32" and out.size == self.nSegments()

        if permanenceThreshold is None:
            self._computeActivity(activeInputs, out)
        else:
            self._permanenceThresholdedComputeActivity(activeInputs,
                                                       permanenceThreshold,
                                                       out)

        return out

    def adjustSynapses(self, segments, activeInputs, activePermanenceDelta,
                       inactivePermanenceDelta):
        self._adjustSynapses(np.asarray(segments, dtype="uint32"),
                             np.asarray(activeInputs, dtype="uint32"),
                             activePermanenceDelta, inactivePermanenceDelta)

    def adjustActiveSynapses(self, segments, activeInputs, permanenceDelta):
        self._adjustActiveSynapses(np.asarray(segments, dtype="uint32"),
                                   np.asarray(activeInputs, dtype="uint32"),
                                   permanenceDelta)

    def adjustInactiveSynapses(self, segments, activeInputs, permanenceDelta):
        self._adjustInactiveSynapses(np.asarray(segments, dtype="uint32"),
                                     np.asarray(activeInputs, dtype="uint32"),
                                     permanenceDelta)

    def growSynapses(self, segments, activeInputs, initialPermanence,
                     assumeInputsSorted=False):
        if not assumeInputsSorted:
            activeInputs = np.sort(activeInputs)

            self._growSynapses(
                np.asarray(segments, dtype="uint32"),
                np.asarray(activeInputs, dtype="uint32"),
                initialPermanence)

    def growSynapsesToSample(self, segments, activeInputs, sampleSize,
                             initialPermanence, rng, assumeInputsSorted=False):
        if not assumeInputsSorted:
            activeInputs = np.sort(activeInputs)

            if isinstance(sampleSize, numbers.Number):
                self._growSynapsesToSample_singleCount(
                    np.asarray(segments, dtype="uint32"),
                    np.asarray(activeInputs, dtype="uint32"),
                    sampleSize,
                    initialPermanence,
                    rng)
            else:
                self._growSynapsesToSample_multipleCounts(
                    np.asarray(segments, dtype="uint32"),
                    np.asarray(activeInputs, dtype="uint32"),
                    np.asarray(sampleSize, dtype="int32"),
                    initialPermanence,
                    rng)

    def clipPermanences(self, segments):
        self._clipPermanences(np.asarray(segments, dtype="uint32"))

    def mapSegmentsToSynapseCounts(self, segments):
        return self._mapSegmentsToSynapseCounts(
            np.asarray(segments, dtype="uint32"))

    def createSegments(self, segments):
        return self._createSegments(np.asarray(segments, dtype="uint32"))

    def destroySegments(self, segments):
        self._destroySegments(np.asarray(segments, dtype="uint32"))

    def getSegmentCounts(self, cells):
        return self._getSegmentCounts(np.asarray(cells, dtype="uint32"))

    def getSegmentsForCell(self, cell):
        return self._getSegmentsForCell(cell)

    def sortSegmentsByCell(self, segments):
        # Can't convert it, since we're sorting it in place.
        assert segments.dtype == "uint32"
        self._sortSegmentsByCell(segments)

    def filterSegmentsByCell(self, segments, cells, assumeSorted=False):
        segments = np.asarray(segments, dtype="uint32")
        cells = np.asarray(cells, dtype="uint32")

        if not assumeSorted:
            segments = np.copy(segments)
            self.sortSegmentsByCell(segments)
            cells = np.sort(cells)

        return self._filterSegmentsByCell(segments, cells)

    def mapSegmentsToCells(self, segments):
        segments = np.asarray(segments, dtype="uint32")
        return self._mapSegmentsToCells(segments)
