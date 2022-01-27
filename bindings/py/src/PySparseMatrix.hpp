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

#ifndef NTA_PY_SPARSE_MATRIX
#define NTA_PY_SPARSE_MATRIX

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <nupic/math/SparseMatrix.hpp>
#include <nupic/utils/Random.hpp>

#include <nupic_module.hpp>
#include "support/PyCapnp.hpp"
#include "support/pybind_helpers.hpp"

namespace nupic {
namespace py_sparse_matrix {

using nupic::Int32;
using nupic::UInt32;
using nupic::UInt64;
using nupic::Real32;

namespace py = pybind11;


void add_to(py::module &m) {

  typedef nupic::SparseMatrix<UInt32, Real32, Int32, Real32, nupic::DistanceToZero<Real32>> SparseMatrix32;

  py::class_<SparseMatrix32>(m, "SparseMatrix")
    .def(py::init<>())
    .def(py::init<UInt32, UInt32>())
    .def("nRows", &SparseMatrix32::nRows)
    .def("nCols", &SparseMatrix32::nCols)
    .def("resize", &SparseMatrix32::resize)
    .def("threshold",
         static_cast<void (SparseMatrix32::*)(const Real32&)>(&SparseMatrix32::threshold),
         "", py::arg("threshold") = nupic::Epsilon)
    .def("normalize", &SparseMatrix32::normalize, "", py::arg("val") = 1.0,
         py::arg("exact") = false)
    .def("get", &SparseMatrix32::get)
    .def("set", &SparseMatrix32::set)
    .def("setRowToZero", &SparseMatrix32::setRowToZero)
    .def("setColToZero", &SparseMatrix32::setColToZero)
    .def("CSRSize", &SparseMatrix32::CSRSize)
    .def("toPyString", [](SparseMatrix32 &self) {
      std::stringstream ss;
      self.toCSR(ss);
      return ss.str();
    })
    .def("clip", &SparseMatrix32::clip)
    .def("add", static_cast<void (SparseMatrix32::*)(const SparseMatrix32&)>(&SparseMatrix32::add))
    .def("addRows", [](SparseMatrix32 &self, py::array_t<UInt32> indicator) {
      py::array_t<Real32> result(self.nCols());
      self.addRows(arr_begin(indicator), arr_end(indicator),
                   arr_begin(result), arr_end(result));
      return result;
    })
    .def("addTwoRows", &SparseMatrix32::addTwoRows)
    .def("getRow", [](SparseMatrix32 &self, UInt32 row) {
      py::array_t<Int32> out(self.nCols());
      self.getRowToDense(row, arr_begin(out));
      return out;
    })
    .def("getCol", [](SparseMatrix32 &self, UInt32 col) {
      py::array_t<Int32> out(self.nRows());
      self.getColToDense(col, arr_begin(out));
      return out;
    })
    .def("toDense", [](SparseMatrix32 &self) {
      py::array_t<Real32> out({self.nRows(), self.nCols()});
      self.toDense(arr_begin(out));
      return out;
    })
    .def("_fromDense", [](SparseMatrix32 &self, UInt32 nrows, UInt32 ncols,
                          py::array_t<Real32> matrix) {
      self.fromDense(nrows, ncols, arr_begin(matrix));
    })
    .def("setRowFromDense", [](SparseMatrix32 &self, UInt32 row,
                               py::array_t<Real32> x) {
      self.setRowFromDense(row, arr_begin(x));
    })
    .def("_writeAsCapnpPyBytes", [](SparseMatrix32 &self) {
      return nupic::PyCapnpHelper::writeAsPyBytes(self);
    })
    .def("_initFromCapnpPyBytes", [](SparseMatrix32 &self, PyObject* pyBytes) {
      nupic::PyCapnpHelper::initFromPyBytes(self, pyBytes);
    })
    .def("_incrementNonZerosOnOuter", [](SparseMatrix32 &self,
                                         py::array_t<UInt32> rows,
                                         py::array_t<UInt32> cols,
                                         Real32 delta) {
      self.incrementNonZerosOnOuter(arr_begin(rows), arr_end(rows),
                                    arr_begin(cols), arr_end(cols),
                                    delta);
    })
    .def("_incrementNonZerosOnRowsExcludingCols", [](SparseMatrix32 &self,
                                                     py::array_t<UInt32> rows,
                                                     py::array_t<UInt32> cols,
                                                     Real32 delta) {
      self.incrementNonZerosOnRowsExcludingCols(arr_begin(rows), arr_end(rows),
                                                arr_begin(cols), arr_end(cols),
                                                delta);
    })
    .def("_setZerosOnOuter", [](SparseMatrix32 &self,
                                py::array_t<UInt32> rows,
                                py::array_t<UInt32> cols,
                                Real32 value) {
      self.setZerosOnOuter(arr_begin(rows), arr_end(rows),
                           arr_begin(cols), arr_end(cols),
                           value);
    })
    .def("_setRandomZerosOnOuter_singleCount", [](SparseMatrix32 &self,
                                                  py::array_t<UInt32> rows,
                                                  py::array_t<UInt32> cols,
                                                  Int32 numNewNonZeros,
                                                  Real32 value,
                                                  nupic::Random& rng) {
      self.setRandomZerosOnOuter(arr_begin(rows), arr_end(rows),
                                 arr_begin(cols), arr_end(cols),
                                 numNewNonZeros, value, rng);
    })
    .def("_setRandomZerosOnOuter_multipleCounts", [](SparseMatrix32 &self,
                                                     py::array_t<UInt32> rows,
                                                     py::array_t<UInt32> cols,
                                                     py::array_t<Int32> numNewNonZeros,
                                                     Real32 value,
                                                     nupic::Random& rng) {
      self.setRandomZerosOnOuter(arr_begin(rows), arr_end(rows),
                                 arr_begin(cols), arr_end(cols),
                                 arr_begin(numNewNonZeros), arr_end(numNewNonZeros),
                                 value, rng);
    })
    .def("_increaseRowNonZeroCountsOnOuterTo", [](SparseMatrix32 &self,
                                                  py::array_t<UInt32> rows,
                                                  py::array_t<UInt32> cols,
                                                  Int32 numDesiredNonZeros,
                                                  Real32 initialValue,
                                                  nupic::Random& rng) {
      self.increaseRowNonZeroCountsOnOuterTo(arr_begin(rows), arr_end(rows),
                                             arr_begin(cols), arr_end(cols),
                                             numDesiredNonZeros, initialValue,
                                             rng);
    })
    .def("_clipRowsBelowAndAbove", [](SparseMatrix32 &self, py::array_t<UInt32> rows,
                                      Real32 a, Real32 b) {
      self.clipRowsBelowAndAbove(arr_begin(rows), arr_end(rows), a, b);
    })
    .def("_nNonZerosPerRow_allRows", [](SparseMatrix32 &self) {
      py::array_t<UInt32> out(self.nRows());
      self.nNonZerosPerRow(arr_begin(out));
      return out;
    })
    .def("_nNonZerosPerRow_allRows", [](SparseMatrix32 &self,
                                        py::array_t<UInt32> rows) {
      py::array_t<UInt32> out(self.nRows());
      self.nNonZerosPerRow(arr_begin(rows), arr_end(rows), arr_begin(out));
      return out;
    })
    .def("_nNonZerosPerRowOnCols", [](SparseMatrix32 &self,
                                      py::array_t<UInt32> rows,
                                      py::array_t<UInt32> cols) {
      py::array_t<UInt32> out(self.nRows());
      self.nNonZerosPerRowOnCols(arr_begin(rows), arr_end(rows),
                                 arr_begin(cols), arr_end(cols),
                                 arr_begin(out));
      return out;
    })
    .def("_rightVecSumAtNZ", [](SparseMatrix32 &self,
                                py::array_t<UInt32> denseArray,
                                py::array_t<Real32> out) {
      self.rightVecSumAtNZ(arr_begin(denseArray), arr_begin(out));
    })
    .def("_rightVecSumAtNZSparse", [](SparseMatrix32 &self,
                                      py::array_t<UInt32> sparseBinaryArray,
                                      py::array_t<Int32> out) {
      self.rightVecSumAtNZSparse(arr_begin(sparseBinaryArray),
                                 arr_end(sparseBinaryArray),
                                 arr_begin(out));
    })
    .def("_rightVecSumAtNZGtThreshold", [](SparseMatrix32 &self,
                                           py::array_t<UInt32> denseArray,
                                           Real32 threshold,
                                           py::array_t<Real32> out) {
      self.rightVecSumAtNZGtThreshold(arr_begin(denseArray),
                                      arr_begin(out), threshold);
    })
    .def("_rightVecSumAtNZGtThresholdSparse", [](SparseMatrix32 &self,
                                                 py::array_t<UInt32> sparseBinaryArray,
                                                 Real32 threshold,
                                                 py::array_t<Int32> out) {
      self.rightVecSumAtNZGtThresholdSparse(arr_begin(sparseBinaryArray),
                                            arr_end(sparseBinaryArray),
                                            arr_begin(out), threshold);
    })
    .def("_rightVecSumAtNZGteThreshold", [](SparseMatrix32 &self,
                                            py::array_t<UInt32> denseArray,
                                            Real32 threshold,
                                            py::array_t<Real32> out) {
      self.rightVecSumAtNZGteThreshold(arr_begin(denseArray),
                                       arr_begin(out), threshold);
    })
    .def("_rightVecSumAtNZGteThresholdSparse", [](SparseMatrix32 &self,
                                                  py::array_t<UInt32> sparseBinaryArray,
                                                  Real32 threshold,
                                                  py::array_t<Int32> out) {
      self.rightVecSumAtNZGteThresholdSparse(arr_begin(sparseBinaryArray),
                                             arr_end(sparseBinaryArray),
                                             arr_begin(out), threshold);
    });
}

} // namespace py_sparse_matrix

} // namespace nupic

#endif  // NTA_PY_SPARSE_MATRIX
