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


class SegmentMatrixAdapterMixin:
    """
    Methods for the SegmentMatrixAdapter<SparseMatrix> C++ class
    """
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
