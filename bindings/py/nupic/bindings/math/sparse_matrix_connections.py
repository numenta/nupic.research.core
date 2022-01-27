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

from .segment_matrix_adapter import SegmentMatrixAdapterMixin


class SparseMatrixConnections(SegmentMatrixAdapterMixin,
                              _nupic.SparseMatrixConnections):
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
