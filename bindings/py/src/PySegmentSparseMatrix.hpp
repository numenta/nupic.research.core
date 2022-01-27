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

#ifndef NTA_PY_SEGMENT_SPARSE_MATRIX
#define NTA_PY_SEGMENT_SPARSE_MATRIX

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <nupic/math/SegmentMatrixAdapter.hpp>
#include <nupic/math/SparseMatrix.hpp>

#include <nupic_module.hpp>
#include "support/PyCapnp.hpp"
#include "support/pybind_helpers.hpp"

namespace nupic {
namespace py_segment_sparse_matrix {

using nupic::Int32;
using nupic::UInt32;
using nupic::UInt64;
using nupic::Real32;
using nupic::Real64;

namespace py = pybind11;


void add_to(py::module &m) {

  typedef nupic::SparseMatrix<UInt32, Real32, Int32, Real64, nupic::DistanceToZero<Real32>> SparseMatrix32;
  typedef SegmentMatrixAdapter<SparseMatrix32> SegmentSparseMatrix;

  py::class_<SegmentSparseMatrix>(m, "SegmentSparseMatrix")
    .def(py::init<UInt32, UInt32>())
    .def_readwrite("matrix", &SegmentSparseMatrix::matrix)
    .def("nCells", &SegmentSparseMatrix::nCells)
    .def("nSegments", &SegmentSparseMatrix::nSegments)
    .def("createSegment", &SegmentSparseMatrix::createSegment)
    .def("_createSegments", [](SegmentSparseMatrix &self, py::array_t<UInt32> cells) {
      py::array_t<UInt32> segments(cells.size());
      self.createSegments(arr_begin(cells), arr_end(cells), arr_begin(segments));
      return segments;
    })
    .def("_destroySegments", [](SegmentSparseMatrix &self,
                                py::array_t<UInt32> segments) {
      self.destroySegments(arr_begin(segments), arr_end(segments));
    })
    .def("_getSegmentCounts", [](SegmentSparseMatrix &self, py::array_t<UInt32> cells) {
      py::array_t<Int32> counts(cells.size());
      self.getSegmentCounts(arr_begin(cells), arr_end(cells), arr_begin(counts));
      return counts;
    })
    .def("_getSegmentsForCell", [](SegmentSparseMatrix &self, UInt32 cell) {
      return py::array_t<UInt32>(
        py::cast(self.getSegmentsForCell(cell)));
    })
    .def("_sortSegmentsByCell", [](SegmentSparseMatrix &self,
                                   py::array_t<UInt32> segments) {
      self.sortSegmentsByCell(arr_begin(segments), arr_end(segments));
    })
    .def("_filterSegmentsByCell", [](SegmentSparseMatrix &self,
                                     py::array_t<UInt32> segments,
                                     py::array_t<UInt32> cells) {
      return py::array_t<UInt32>(
        py::cast(
          self.filterSegmentsByCell(arr_begin(segments), arr_end(segments),
                                    arr_begin(cells), arr_end(cells))));
    })
    .def("_mapSegmentsToCells", [](SegmentSparseMatrix &self, py::array_t<UInt32> segments) {
      py::array_t<UInt32> cells(segments.size());
      self.mapSegmentsToCells(arr_begin(segments), arr_end(segments), arr_begin(cells));
      return cells;
    });
}

} // namespace py_segment_sparse_matrix

} // namespace nupic

#endif  // NTA_PY_SEGMENT_SPARSE_MATRIX
