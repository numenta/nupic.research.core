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

#ifndef NTA_PY_CONNECTIONS
#define NTA_PY_CONNECTIONS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <vector>

#include <nupic/algorithms/Connections.hpp>

#include <nupic_module.hpp>
#include "support/pybind_helpers.hpp"

namespace nupic {
namespace py_connections {

using nupic::Int;
using nupic::UInt;
using namespace nupic::algorithms::connections;

namespace py = pybind11;


void add_to(py::module &m) {
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
}

} // namespace py_connections
} // namespace nupic

#endif // NTA_PY_CONNECTIONS
