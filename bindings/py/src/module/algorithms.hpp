/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------------
 */

#ifndef NTA_PYBIND_ALGORITHMS
#define NTA_PYBIND_ALGORITHMS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <vector>

#include <nupic/algorithms/ApicalTiebreakTemporalMemory.hpp>

#include "support/PyCapnp.hpp"
#include "support/pybind_helpers.hpp"

using nupic::Int;
using nupic::UInt;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::apical_tiebreak_temporal_memory;

namespace py = pybind11;


void module_add_algorithms(py::module &m) {
  py::class_<Synapse>(m, "Synapse");
  py::class_<SynapseData>(m, "SynapseData")
    .def_readwrite("presynapticCell", &SynapseData::presynapticCell)
    .def_readwrite("permanence", &SynapseData::permanence)
    .def_readwrite("segment", &SynapseData::segment);
  py::class_<SegmentData>(m, "SegmentData")
    .def_readwrite("synapses",&SegmentData::synapses)
    .def_readwrite("cell",&SegmentData::cell);
  py::class_<CellData>(m, "CellData")
    .def_readwrite("segments", &CellData::segments);

  py::class_<Connections>(m, "Connections")
    .def(py::init<>())
    .def(py::init<CellIdx>())
    .def("initialize", &Connections::initialize)
    .def("createSegment", &Connections::createSegment)
    .def("createSynapse", &Connections::createSynapse)
    .def("destroySegment", &Connections::destroySegment)
    .def("destroySynapse", &Connections::destroySynapse)
    .def("updateSynapsePermanence", &Connections::updateSynapsePermanence)
    .def("segmentsForCell", &Connections::segmentsForCell)
    .def("synapsesForSegment", &Connections::synapsesForSegment)
    .def("cellForSegment", &Connections::cellForSegment)
    .def("idxOnCellForSegment", &Connections::idxOnCellForSegment)
    .def("mapSegmentsToCells", [](Connections &self, py::array_t<Segment> segments) {
      py::array_t<CellIdx> cells(segments.size());
      self.mapSegmentsToCells(arr_begin(segments), arr_end(segments), arr_begin(cells));
      return cells;
    })
    .def("segmentForSynapse", &Connections::segmentForSynapse)
    .def("dataForSegment", &Connections::dataForSegment)
    .def("dataForSynapse", &Connections::dataForSynapse)
    .def("getSegment", &Connections::getSegment)
    .def("segmentFlatListLength", &Connections::segmentFlatListLength)
    .def("compareSegments", &Connections::compareSegments)
    .def("synapsesForPresynapticCell", &Connections::synapsesForPresynapticCell)
    .def("numCells", &Connections::numCells)
    .def("numSegments", static_cast<UInt (Connections::*)() const>(&Connections::numSegments))
    .def("numSegments", static_cast<UInt (Connections::*)(CellIdx) const>(&Connections::numSegments))
    .def("numSynapses", static_cast<UInt (Connections::*)() const>(&Connections::numSynapses))
    .def("numSynapses", static_cast<UInt (Connections::*)(Segment) const>(&Connections::numSynapses));

  py::class_<ApicalTiebreakTemporalMemory>(m, "ApicalTiebreakTemporalMemory")
    .def(py::init<>())
    .def(py::init<UInt, UInt, UInt, UInt, UInt, Permanence, Permanence, UInt,
                  UInt, Permanence, Permanence, Permanence, Permanence, bool,
                  Int, UInt, UInt, bool>())
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def_readwrite("basalConnections", &ApicalTiebreakTemporalMemory::basalConnections)
    .def_readwrite("apicalConnections", &ApicalTiebreakTemporalMemory::apicalConnections)
    .def("seed", &ApicalTiebreakTemporalMemory::seed)
    .def("reset", &ApicalTiebreakTemporalMemory::reset)
    .def("createBasalSegment", &ApicalTiebreakTemporalMemory::createBasalSegment)
    .def("createApicalSegment", &ApicalTiebreakTemporalMemory::createApicalSegment)
    .def("numberOfCells", &ApicalTiebreakTemporalMemory::numberOfCells)
    .def("getActiveCells", [](ApicalTiebreakTemporalMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getActiveCells())
        );
    })
    .def("getPredictedCells", [](ApicalTiebreakTemporalMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getPredictedCells())
        );
    })
    .def("getPredictedActiveCells", [](ApicalTiebreakTemporalMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getPredictedActiveCells())
        );
    })
    .def("getWinnerCells", [](ApicalTiebreakTemporalMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getWinnerCells())
        );
    })
    .def("getBasalInputSize", &ApicalTiebreakTemporalMemory::getBasalInputSize)
    .def("getApicalInputSize", &ApicalTiebreakTemporalMemory::getApicalInputSize)
    .def("numberOfColumns", &ApicalTiebreakTemporalMemory::numberOfColumns)
    .def("getCellsPerColumn", &ApicalTiebreakTemporalMemory::getCellsPerColumn)
    .def("getActivationThreshold", &ApicalTiebreakTemporalMemory::getActivationThreshold)
    .def("setActivationThreshold", &ApicalTiebreakTemporalMemory::setActivationThreshold)
    .def("getInitialPermanence", &ApicalTiebreakTemporalMemory::getInitialPermanence)
    .def("setInitialPermanence", &ApicalTiebreakTemporalMemory::setInitialPermanence)
    .def("getConnectedPermanence", &ApicalTiebreakTemporalMemory::getConnectedPermanence)
    .def("setConnectedPermanence", &ApicalTiebreakTemporalMemory::setConnectedPermanence)
    .def("getMinThreshold", &ApicalTiebreakTemporalMemory::getMinThreshold)
    .def("setMinThreshold", &ApicalTiebreakTemporalMemory::setMinThreshold)
    .def("getSampleSize", &ApicalTiebreakTemporalMemory::getSampleSize)
    .def("setSampleSize", &ApicalTiebreakTemporalMemory::setSampleSize)
    .def("getPermanenceIncrement", &ApicalTiebreakTemporalMemory::getPermanenceIncrement)
    .def("setPermanenceIncrement", &ApicalTiebreakTemporalMemory::setPermanenceIncrement)
    .def("getPermanenceDecrement", &ApicalTiebreakTemporalMemory::getPermanenceDecrement)
    .def("setPermanenceDecrement", &ApicalTiebreakTemporalMemory::setPermanenceDecrement)
    .def("getBasalPredictedSegmentDecrement", &ApicalTiebreakTemporalMemory::getBasalPredictedSegmentDecrement)
    .def("setBasalPredictedSegmentDecrement", &ApicalTiebreakTemporalMemory::setBasalPredictedSegmentDecrement)
    .def("getApicalPredictedSegmentDecrement", &ApicalTiebreakTemporalMemory::getApicalPredictedSegmentDecrement)
    .def("setApicalPredictedSegmentDecrement", &ApicalTiebreakTemporalMemory::setApicalPredictedSegmentDecrement)
    .def("getMaxSegmentsPerCell", &ApicalTiebreakTemporalMemory::getMaxSegmentsPerCell)
    .def("getMaxSynapsesPerSegment", &ApicalTiebreakTemporalMemory::getMaxSynapsesPerSegment)
    .def("getCheckInputs", &ApicalTiebreakTemporalMemory::getCheckInputs)
    .def("setCheckInputs", &ApicalTiebreakTemporalMemory::setCheckInputs)
    // .def("write", &ApicalTiebreakTemporalMemory::write)
    // .def("read", &ApicalTiebreakTemporalMemory::read)
    .def("printParameters", &ApicalTiebreakTemporalMemory::printParameters)
    .def("columnForCell", &ApicalTiebreakTemporalMemory::columnForCell);


  py::class_<ApicalTiebreakPairMemory, ApicalTiebreakTemporalMemory>(m, "ApicalTiebreakPairMemory")
    .def(py::init<UInt, UInt, UInt, UInt, UInt, Permanence, Permanence, UInt,
                  UInt, Permanence, Permanence, Permanence, Permanence, bool,
                  Int, UInt, UInt, bool>())
    .def("_compute", [](ApicalTiebreakPairMemory &self, py::array_t<UInt> activeColumns,
                        py::array_t<UInt> basalInput, py::array_t<UInt> apicalInput,
                        py::array_t<UInt> basalGrowthCandidates,
                        py::array_t<UInt> apicalGrowthCandidates, bool learn) {
      self.compute(arr_begin(activeColumns), arr_end(activeColumns),
                   arr_begin(basalInput), arr_end(basalInput),
                   arr_begin(apicalInput), arr_end(apicalInput),
                   arr_begin(basalGrowthCandidates), arr_end(basalGrowthCandidates),
                   arr_begin(apicalGrowthCandidates), arr_end(apicalGrowthCandidates),
                   learn);
    })
    .def("getBasalPredictedCells", [](ApicalTiebreakPairMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getBasalPredictedCells())
        );
    })
    .def("getApicalPredictedCells", [](ApicalTiebreakPairMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getApicalPredictedCells())
        );
    });

  py::class_<ApicalTiebreakSequenceMemory, ApicalTiebreakTemporalMemory>(m, "ApicalTiebreakSequenceMemory")
    .def(py::init<UInt, UInt, UInt, UInt, Permanence, Permanence, UInt,
                  UInt, Permanence, Permanence, Permanence, Permanence, bool,
                  Int, UInt, UInt, bool>())
    .def("_compute", [](ApicalTiebreakSequenceMemory &self, py::array_t<UInt> activeColumns,
                        py::array_t<UInt> apicalInput, py::array_t<UInt> apicalGrowthCandidates,
                        bool learn) {
      self.compute(arr_begin(activeColumns), arr_end(activeColumns),
                   arr_begin(apicalInput), arr_end(apicalInput),
                   arr_begin(apicalGrowthCandidates), arr_end(apicalGrowthCandidates),
                   learn);
    })
    .def("getNextPredictedCells", [](ApicalTiebreakSequenceMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getNextPredictedCells())
        );
    })
    .def("getNextBasalPredictedCells", [](ApicalTiebreakSequenceMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getNextBasalPredictedCells())
        );
    })
    .def("getNextApicalPredictedCells", [](ApicalTiebreakSequenceMemory &self) {
      return py::array_t<UInt>(
        py::cast(self.getNextApicalPredictedCells())
        );
    });

}

#endif // NTA_PYBIND_ALGORITHMS
