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

#ifndef NTA_PY_SPARSE_MATRIX_CONNECTIONS
#define NTA_PY_SPARSE_MATRIX_CONNECTIONS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <nupic_module.hpp>
#include "nupic/math/SparseMatrixConnections.hpp"
#include "support/pybind_helpers.hpp"

namespace nupic {
namespace py_sparse_matrix_connections {

using nupic::SparseMatrixConnections;
using nupic::Int32;
using nupic::UInt32;
using nupic::UInt64;
using nupic::Real32;
using nupic::Random;

namespace py = pybind11;


void add_to(py::module &m) {
  typedef nupic::SparseMatrix<UInt32, Real32, Int32, Real64, nupic::DistanceToZero<Real32>> SparseMatrix32;
  typedef SegmentMatrixAdapter<SparseMatrix32> SegmentSparseMatrix;

  py::class_<SparseMatrixConnections, SegmentSparseMatrix>(m, "SparseMatrixConnections")
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
    });
}

} // namespace py_sparse_matrix_connections
} // namespace nupic

#endif // NTA_PY_SPARSE_MATRIX_CONNECTIONS
