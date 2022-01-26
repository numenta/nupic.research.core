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


try:
    # NOTE need to import capnp first to activate the magic necessary for
    # ApicalTiebreakTemporalMemory
    import capnp
except ImportError:
    capnp = None
else:
    from nupic.proto.ApicalTiebreakTemporalMemoryProto_capnp import (
        ApicalTiebreakSequenceMemoryProto,
        ApicalTiebreakTemporalMemoryProto,
    )


# Capnp reader traveral limit (see capnp::ReaderOptions)
_TRAVERSAL_LIMIT_IN_WORDS = 1 << 63



class ApicalTiebreakPairMemory(_nupic.ApicalTiebreakPairMemory):
    def __init__(self,
                 columnCount=2048,
                 basalInputSize=0,
                 apicalInputSize=0,
                 cellsPerColumn=32,
                 activationThreshold=13,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 minThreshold=10,
                 sampleSize=20,
                 permanenceIncrement=0.10,
                 permanenceDecrement=0.10,
                 basalPredictedSegmentDecrement=0.00,
                 apicalPredictedSegmentDecrement=0.00,
                 learnOnOneCell=False,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=255,
                 seed=42,
                 checkInputs=True,
                 basalInputPrepend=False):
      """
      @param columnCount (int)
      The number of minicolumns

      @param basalInputSize (sequence)
      The number of bits in the basal input

      @param apicalInputSize (int)
      The number of bits in the apical input

      @param cellsPerColumn (int)
      Number of cells per column

      @param activationThreshold (int)
      If the number of active connected synapses on a segment is at least this
      threshold, the segment is said to be active.

      @param initialPermanence (float)
      Initial permanence of a new synapse

      @param connectedPermanence (float)
      If the permanence value for a synapse is greater than this value, it is said
      to be connected.

      @param minThreshold (int)
      If the number of potential synapses active on a segment is at least this
      threshold, it is said to be "matching" and is eligible for learning.

      @param sampleSize (int)
      How much of the active SDR to sample with synapses.

      @param permanenceIncrement (float)
      Amount by which permanences of synapses are incremented during learning.

      @param permanenceDecrement (float)
      Amount by which permanences of synapses are decremented during learning.

      @param predictedSegmentDecrement (float)
      Amount by which basal segments are punished for incorrect predictions.

      @param learnOnOneCell (bool)
      Whether to always choose the same cell when bursting a column until the
      next reset occurs.

      @param maxSegmentsPerCell (int)
      The maximum number of segments per cell.

      @param maxSynapsesPerSegment (int)
      The maximum number of synapses per segment.

      @param seed (int)
      Seed for the random number generator.

      @param basalInputPrepend (bool)
      If true, this TM will automatically insert its activeCells and winnerCells
      into the basalInput and basalGrowthCandidates, respectively.
      """

      if basalInputPrepend:
        basalInputSize += columnCount * cellsPerColumn

      super().__init__(
          columnCount, basalInputSize, apicalInputSize,
          cellsPerColumn, activationThreshold,
          initialPermanence, connectedPermanence,
          minThreshold, sampleSize, permanenceIncrement,
          permanenceDecrement, basalPredictedSegmentDecrement,
          apicalPredictedSegmentDecrement,
          learnOnOneCell, seed, maxSegmentsPerCell,
          maxSynapsesPerSegment, checkInputs)

      self.basalInputPrepend = basalInputPrepend


    # def __getstate__(self):
    #   # Save the local attributes but override the C++ temporal memory with the
    #   # string representation.
    #   d = dict(self.__dict__)
    #   d["this"] = self.getCState()
    #   return d


    # def __setstate__(self, state):
    #   if isinstance(state, str):
    #     self.loadFromString(state)
    #     self.valueToCategory = {}
    #   else:
    #     self.loadFromString(state["this"])
    #     # Use the rest of the state to set local Python attributes.
    #     del state["this"]
    #     self.__dict__.update(state)



    def compute(self,
                activeColumns,
                basalInput=(),
                apicalInput=(),
                basalGrowthCandidates=None,
                apicalGrowthCandidates=None,
                learn=True):
      """
      Perform one time step of the Temporal Memory algorithm.

      @param activeColumns (sequence)
      Sorted list of active columns.

      @param basalInput (sequence)
      Sorted list of active input bits for the basal dendrite segments.

      @param apicalInput (sequence)
      Sorted list of active input bits for the apical dendrite segments

      @param basalGrowthCandidates (sequence)
      List of bits that the active cells may grow new basal synapses to.
      If None, the basalInput is assumed to be growth candidates.

      @param apicalGrowthCandidates (sequence)
      List of bits that the active cells may grow new apical synapses to
      If None, the apicalInput is assumed to be growth candidates.

      @param learn (bool)
      Whether or not learning is enabled
      """

      npBasal = np.asarray(basalInput, "uint32")
      npApical = np.asarray(apicalInput, "uint32")
      npBasalGrowth = (np.asarray(basalGrowthCandidates, "uint32")
                       if basalGrowthCandidates is not None
                       else npBasal)
      npApicalGrowth = (np.asarray(apicalGrowthCandidates, "uint32")
                        if apicalGrowthCandidates is not None
                        else npApical)

      if self.basalInputPrepend:
        npBasal = np.append(self.getActiveCells(),
                               npBasal + self.numberOfCells())
        npBasalGrowth = np.append(self.getWinnerCells(),
                                     npBasalGrowth + self.numberOfCells())

      self._compute(np.asarray(activeColumns, "uint32"),
                    npBasal, npApical, npBasalGrowth, npApicalGrowth,
                    learn)


    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance

    def write(self, pyBuilder):
      """Serialize the ApicalTiebreakTemporalMemory instance using capnp.

      :param: Destination ApicalTiebreakTemporalMemoryProto message builder
      """
      reader = ApicalTiebreakTemporalMemoryProto.from_bytes(
        self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy


    def convertedRead(self, proto):
      """Initialize the ApicalTiebreakTemporalMemory instance from the given
      ApicalTiebreakTemporalMemoryProto reader.

      :param proto: ApicalTiebreakTemporalMemoryProto message reader containing data
                    from a previously serialized ApicalTiebreakTemporalMemory
                    instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2


class ApicalTiebreakSequenceMemory(_nupic.ApicalTiebreakSequenceMemory):
    def __init__(self,
                 columnCount=2048,
                 apicalInputSize=0,
                 cellsPerColumn=32,
                 activationThreshold=13,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 minThreshold=10,
                 sampleSize=20,
                 permanenceIncrement=0.10,
                 permanenceDecrement=0.10,
                 basalPredictedSegmentDecrement=0.00,
                 apicalPredictedSegmentDecrement=0.00,
                 learnOnOneCell=False,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=255,
                 seed=42,
                 checkInputs=True,
                 basalInputPrepend=False):
      """
      @param columnCount (int)
      The number of minicolumns

      @param apicalInputSize (int)
      The number of bits in the apical input

      @param cellsPerColumn (int)
      Number of cells per column

      @param activationThreshold (int)
      If the number of active connected synapses on a segment is at least this
      threshold, the segment is said to be active.

      @param initialPermanence (float)
      Initial permanence of a new synapse

      @param connectedPermanence (float)
      If the permanence value for a synapse is greater than this value, it is said
      to be connected.

      @param minThreshold (int)
      If the number of potential synapses active on a segment is at least this
      threshold, it is said to be "matching" and is eligible for learning.

      @param sampleSize (int)
      How much of the active SDR to sample with synapses.

      @param permanenceIncrement (float)
      Amount by which permanences of synapses are incremented during learning.

      @param permanenceDecrement (float)
      Amount by which permanences of synapses are decremented during learning.

      @param predictedSegmentDecrement (float)
      Amount by which basal segments are punished for incorrect predictions.

      @param learnOnOneCell (bool)
      Whether to always choose the same cell when bursting a column until the
      next reset occurs.

      @param maxSegmentsPerCell (int)
      The maximum number of segments per cell.

      @param maxSynapsesPerSegment (int)
      The maximum number of synapses per segment.

      @param seed (int)
      Seed for the random number generator.

      @param basalInputPrepend (bool)
      If true, this TM will automatically insert its activeCells and winnerCells
      into the basalInput and basalGrowthCandidates, respectively.
      """

      super().__init__(
        columnCount, apicalInputSize,
        cellsPerColumn, activationThreshold,
        initialPermanence, connectedPermanence,
        minThreshold, sampleSize, permanenceIncrement,
        permanenceDecrement, basalPredictedSegmentDecrement,
        apicalPredictedSegmentDecrement,
        learnOnOneCell, seed, maxSegmentsPerCell,
        maxSynapsesPerSegment, checkInputs)


    # def __getstate__(self):
    #   # Save the local attributes but override the C++ temporal memory with the
    #   # string representation.
    #   d = dict(self.__dict__)
    #   d["this"] = self.getCState()
    #   return d


    # def __setstate__(self, state):
    #   # Create an empty C++ temporal memory and populate it from the serialized
    #   # string.
    #   self.this = _EXPERIMENTAL.new_ApicalTiebreakSequenceMemory()
    #   if isinstance(state, str):
    #     self.loadFromString(state)
    #     self.valueToCategory = {}
    #   else:
    #     self.loadFromString(state["this"])
    #     # Use the rest of the state to set local Python attributes.
    #     del state["this"]
    #     self.__dict__.update(state)


    def compute(self,
                activeColumns,
                apicalInput=(),
                apicalGrowthCandidates=None,
                learn=True):
      """
      Perform one time step of the Temporal Memory algorithm.

      @param activeColumns (sequence)
      Sorted list of active columns.

      @param apicalInput (sequence)
      Sorted list of active input bits for the apical dendrite segments

      @param apicalGrowthCandidates (sequence)
      List of bits that the active cells may grow new apical synapses to
      If None, the apicalInput is assumed to be growth candidates.

      @param learn (bool)
      Whether or not learning is enabled
      """

      npApical = np.asarray(apicalInput, "uint32")
      npApicalGrowth = (np.asarray(apicalGrowthCandidates, "uint32")
                        if apicalGrowthCandidates is not None
                        else npApical)

      self._compute(np.asarray(activeColumns, "uint32"),
                    npApical, npApicalGrowth,
                    learn)


    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance

    def write(self, pyBuilder):
      """Serialize the ApicalTiebreakTemporalMemory instance using capnp.

      :param: Destination ApicalTiebreakSequenceMemoryProto message builder
      """
      reader = ApicalTiebreakSequenceMemoryProto.from_bytes(
        self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy


    def convertedRead(self, proto):
      """Initialize the ApicalTiebreakTemporalMemory instance from the given
      ApicalTiebreakSequenceMemoryProto reader.

      :param proto: ApicalTiebreakSequenceMemoryProto message reader containing data
                    from a previously serialized ApicalTiebreakTemporalMemory
                    instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2
