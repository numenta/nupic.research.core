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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

#include "nupic/math/SparseMatrixConnections.hpp"
#include "nupic/utils/Random.hpp"

using std::pair;
using std::vector;
using nupic::SparseMatrixConnections;
using nupic::Int32;
using nupic::UInt32;
using nupic::UInt64;
using nupic::Real32;
using nupic::Random;

namespace py = pybind11;


template <typename T>
T* arr_begin(py::array_t<T> &arr)
{
  return (T*)arr.request().ptr;
}


template <typename T>
T* arr_end(py::array_t<T> &arr)
{
  return (T*)arr_begin(arr) + arr.size();
}


PYBIND11_MODULE(_nupic, m)
{
  py::class_<Random>(m, "Random")
    .def(py::init<>())
    .def(py::init<UInt64>())
    .def("getUInt32", &Random::getUInt32)
    .def("initializeReal32Array", [](Random &self, py::array_t<Real32> array) {
      for (auto it = arr_begin(array); it != arr_end(array); ++it) {
        *it = self.getReal64();
      }
    });

  py::class_<SparseMatrixConnections>(m, "SparseMatrixConnections")
    .def(py::init<UInt32, UInt32>())
    .def("_computeActivity", [](SparseMatrixConnections &self,
                                py::array_t<UInt32> activeInputs,
                                py::array_t<Int32> overlaps) {
      self.computeActivity(arr_begin(activeInputs), arr_end(activeInputs),
                           arr_begin(overlaps));
    })
    .def("_permanenceThresholdedComputeActivity", [](SparseMatrixConnections &self,
                                                     py::array_t<UInt32> activeInputs,
                                                     Real32 permanenceThreshold,
                                                     py::array_t<Int32> overlaps) {
      self.computeActivity(arr_begin(activeInputs), arr_end(activeInputs),
                           permanenceThreshold, arr_begin(overlaps));
    })
    .def("_adjustSynapses", [](SparseMatrixConnections &self,
                               py::array_t<UInt32> segments,
                               py::array_t<UInt32> activeInputs,
                               Real32 activePermanenceDelta,
                               Real32 inactivePermanenceDelta) {
      self.adjustSynapses(arr_begin(segments), arr_end(segments),
                          arr_begin(activeInputs), arr_end(activeInputs),
                          activePermanenceDelta, inactivePermanenceDelta);
    })
    .def("_adjustActiveSynapses", [](SparseMatrixConnections &self,
                                     py::array_t<UInt32> segments,
                                     py::array_t<UInt32> activeInputs,
                                     Real32 permanenceDelta) {
      self.adjustActiveSynapses(arr_begin(segments), arr_end(segments),
                                arr_begin(activeInputs), arr_end(activeInputs),
                                permanenceDelta);
    })
    .def("_adjustInactiveSynapses", [](SparseMatrixConnections &self,
                                       py::array_t<UInt32> segments,
                                       py::array_t<UInt32> activeInputs,
                                       Real32 permanenceDelta) {
      self.adjustInactiveSynapses(arr_begin(segments), arr_end(segments),
                                  arr_begin(activeInputs), arr_end(activeInputs),
                                  permanenceDelta);
    })
    .def("_growSynapses", [](SparseMatrixConnections &self,
                             py::array_t<UInt32> segments,
                             py::array_t<UInt32> activeInputs,
                             Real32 initialPermanence) {
      self.growSynapses(arr_begin(segments), arr_end(segments),
                        arr_begin(activeInputs), arr_end(activeInputs),
                        initialPermanence);
    })
    .def("_growSynapsesToSample_singleCount", [](
           SparseMatrixConnections &self,
           py::array_t<UInt32> segments,
           py::array_t<UInt32> activeInputs,
           Int32 sampleSize,
           Real32 initialPermanence,
           Random& rng) {
      self.growSynapsesToSample(arr_begin(segments), arr_end(segments),
                                arr_begin(activeInputs), arr_end(activeInputs),
                                sampleSize, initialPermanence, rng);
    })
    .def("_growSynapsesToSample_multipleCounts", [](
           SparseMatrixConnections &self,
           py::array_t<UInt32> segments,
           py::array_t<UInt32> activeInputs,
           py::array_t<Int32> sampleSizes,
           Real32 initialPermanence,
           Random& rng) {
      self.growSynapsesToSample(arr_begin(segments), arr_end(segments),
                                arr_begin(activeInputs), arr_end(activeInputs),
                                arr_begin(sampleSizes), arr_end(sampleSizes),
                                initialPermanence, rng);
    })
    .def("_clipPermanences", [](SparseMatrixConnections &self,
                                py::array_t<UInt32> segments) {
      self.clipPermanences(arr_begin(segments), arr_end(segments));
    })
    .def("_mapSegmentsToSynapseCounts", [](SparseMatrixConnections &self,
                                          py::array_t<UInt32> segments) {
      py::array_t<Int32> out(segments.size());
      self.mapSegmentsToSynapseCounts(arr_begin(segments), arr_end(segments),
                                      arr_begin(out));
      return out;
    })

    //
    // SegmentMatrixAdapter methods
    //
    .def("nCells", &SparseMatrixConnections::nCells)
    .def("nSegments", &SparseMatrixConnections::nSegments)
    .def("_createSegments", [](SparseMatrixConnections &self, py::array_t<UInt32> cells) {
      py::array_t<UInt32> segments(cells.size());
      self.createSegments(arr_begin(cells), arr_end(cells), arr_begin(segments));
      return segments;
    })
    .def("_destroySegments", [](SparseMatrixConnections &self,
                                py::array_t<UInt32> segments) {
      self.destroySegments(arr_begin(segments), arr_end(segments));
    })
    .def("_getSegmentCounts", [](SparseMatrixConnections &self, py::array_t<UInt32> cells) {
      py::array_t<Int32> counts(cells.size());
      self.getSegmentCounts(arr_begin(cells), arr_end(cells), arr_begin(counts));
      return counts;
    })
    .def("_getSegmentsForCell", [](SparseMatrixConnections &self, UInt32 cell) {
      return py::array_t<UInt32>(
        py::cast(self.getSegmentsForCell(cell)));
    })
    .def("_sortSegmentsByCell", [](SparseMatrixConnections &self,
                                   py::array_t<UInt32> segments) {
      self.sortSegmentsByCell(arr_begin(segments), arr_end(segments));
    })
    .def("_filterSegmentsByCell", [](SparseMatrixConnections &self,
                                     py::array_t<UInt32> segments,
                                     py::array_t<UInt32> cells) {
      return py::array_t<UInt32>(
        py::cast(
          self.filterSegmentsByCell(arr_begin(segments), arr_end(segments),
                                    arr_begin(cells), arr_end(cells))));
    })
    .def("_mapSegmentsToCells", [](SparseMatrixConnections &self, py::array_t<UInt32> segments) {
      py::array_t<UInt32> cells(segments.size());
      self.mapSegmentsToCells(arr_begin(segments), arr_end(segments), arr_begin(cells));
      return cells;
    });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
