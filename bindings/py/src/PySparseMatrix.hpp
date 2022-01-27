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
using nupic::Real64;

namespace py = pybind11;

/**
 * Duck type, to act like a Sparse Matrix. Exposes py::array_t<float32>
 * functionality in a way that allows the SparseMatrix templated methods to use
 * it directly.
 */
struct PyArraySparseMatrixMimic {
  PyArraySparseMatrixMimic(py::array_t<Real32> arr)
    : arr_(arr)
  {
    NTA_CHECK(arr_.request().ndim == 2);
  }
  UInt32 nRows() const {
    return arr_.request().shape[0];
  }
  UInt32 nCols() const {
    return arr_.request().shape[1];
  }
  Real32 get(UInt32 i, UInt32 j) const {
    return arr_.at(i, j);
  }

  py::array_t<Real32> arr_;
};

void add_to(py::module &m) {

  typedef nupic::SparseMatrix<UInt32, Real32, Int32, Real64, nupic::DistanceToZero<Real32>> SparseMatrix32;

  py::class_<SparseMatrix32>(m, "SparseMatrix")
    .def(py::init<>())
    .def(py::init<SparseMatrix32>())
    .def(py::init<UInt32, UInt32>())
    .def("nRows", &SparseMatrix32::nRows)
    .def("nCols", &SparseMatrix32::nCols)
    .def("resize", &SparseMatrix32::resize, "",
         py::arg("new_nrows"), py::arg("new_ncols"), py::arg("setToZero") = false)
    .def("reshape", &SparseMatrix32::reshape)
    .def("threshold",
         // Equivalent to py::overload_cast<const Real32&>(&SparseMatrix32::threshold),
         // which doesn't compile for some reason.
         static_cast<void (SparseMatrix32::*)(const Real32&)>(&SparseMatrix32::threshold),
         "", py::arg("threshold") = nupic::Epsilon)
    .def("thresholdRow",
         // Equivalent to py::overload_cast<UInt32, const Real32&>(&SparseMatrix32::thresholdRow),
         // which doesn't compile for some reason.
         static_cast<void (SparseMatrix32::*)(UInt32, const Real32&)>(&SparseMatrix32::thresholdRow),
         "", py::arg("row"), py::arg("threshold") = nupic::Epsilon)
    .def("thresholdCol",
         // Equivalent to py::overload_cast<UInt32, const Real32&>(&SparseMatrix32::thresholdCol),
         // which doesn't compile for some reason.
         static_cast<void (SparseMatrix32::*)(UInt32, const Real32&)>(&SparseMatrix32::thresholdCol),
         "", py::arg("col"), py::arg("threshold") = nupic::Epsilon)
    .def("normalize", &SparseMatrix32::normalize, "", py::arg("val") = 1.0,
         py::arg("exact") = false)
    .def("normalizeRow", &SparseMatrix32::normalizeRow, "", py::arg("row"),
         py::arg("val") = 1.0, py::arg("exact") = false)
    .def("normalizeCol", &SparseMatrix32::normalizeCol, "", py::arg("col"),
         py::arg("val") = 1.0, py::arg("exact") = false)
    .def("get", &SparseMatrix32::get)
    .def("set", &SparseMatrix32::set, "", py::arg("i"), py::arg("j"), py::arg("val"),
         py::arg("resizeYesNo") = false)
    .def("setRowToZero", &SparseMatrix32::setRowToZero)
    .def("setColToZero", &SparseMatrix32::setColToZero)
    .def("CSRSize", &SparseMatrix32::CSRSize)
    .def("isZero", &SparseMatrix32::isZero)
    .def("isRowZero", &SparseMatrix32::isRowZero)
    .def("isColZero", &SparseMatrix32::isColZero)
    .def("nNonZeros", &SparseMatrix32::nNonZeros)
    .def("nNonZerosOnRow", &SparseMatrix32::nNonZerosOnRow)
    .def("rowNonZeros", [](SparseMatrix32 &self, UInt32 row) {
      const UInt32 n = self.nNonZerosOnRow(row);
      py::array_t<UInt32> ind(n);
      py::array_t<Real32> val(n);
      self.getRowToSparse(row, arr_begin(ind), arr_begin(val));
      return py::make_tuple(ind, val);
    })
    .def("nNonZerosOnCol", &SparseMatrix32::nNonZerosOnCol)
    .def("colNonZeros", [](SparseMatrix32 &self, UInt32 col) {
      const UInt32 n = self.nNonZerosOnCol(col);
      py::array_t<UInt32> ind(n);
      py::array_t<Real32> val(n);
      self.getColToSparse(col, arr_begin(ind), arr_begin(val));
      return py::make_tuple(ind, val);
    })
    .def("sum", &SparseMatrix32::sum)
    .def("prod", &SparseMatrix32::prod)
    .def("min", [](SparseMatrix32 &self) {
      UInt32 min_row, min_col;
      Real32 min_val;
      self.min(min_row, min_col, min_val);
      return py::make_tuple(min_row, min_col, min_val);
    })
    .def("max", [](SparseMatrix32 &self) {
      UInt32 max_row, max_col;
      Real32 max_val;
      self.max(max_row, max_col, max_val);
      return py::make_tuple(max_row, max_col, max_val);
    })
    .def("rowMin", [](SparseMatrix32 &self) {
      py::array_t<UInt32> ind(self.nRows());
      py::array_t<Real32> val(self.nRows());
      self.rowMax(arr_begin(ind), arr_begin(val));
      return py::make_tuple(ind, val);
    })
    .def("rowMax", [](SparseMatrix32 &self) {
      py::array_t<UInt32> ind(self.nRows());
      py::array_t<Real32> val(self.nRows());
      self.rowMax(arr_begin(ind), arr_begin(val));
      return py::make_tuple(ind, val);
    })
    .def("rowMin", [](SparseMatrix32 &self, UInt row_index) {
      UInt32 idx;
      Real32 min_val;
      self.rowMin(row_index, idx, min_val);
      return py::make_tuple(idx, min_val);
    })
    .def("rowMax", [](SparseMatrix32 &self, UInt row_index) {
      UInt32 idx;
      Real32 max_val;
      self.rowMax(row_index, idx, max_val);
      return py::make_tuple(idx, max_val);
    })
    .def("rowSum", &SparseMatrix32::rowSum)
    .def("rowSums", [](SparseMatrix32 &self) {
      py::array_t<Real32> out(self.nRows());
      self.rowSums(arr_begin(out));
      return out;
    })
    .def("rowProd", &SparseMatrix32::rowProd)
    .def("rowProds", [](SparseMatrix32 &self) {
      py::array_t<Real32> out(self.nRows());
      self.rowProds(arr_begin(out));
      return out;
    })
    .def("colMin", [](SparseMatrix32 &self) {
      py::array_t<UInt32> ind(self.nCols());
      py::array_t<Real32> val(self.nCols());
      self.colMax(arr_begin(ind), arr_begin(val));
      return py::make_tuple(ind, val);
    })
    .def("colMax", [](SparseMatrix32 &self) {
      py::array_t<UInt32> ind(self.nCols());
      py::array_t<Real32> val(self.nCols());
      self.colMax(arr_begin(ind), arr_begin(val));
      return py::make_tuple(ind, val);
    })
    .def("colMin", [](SparseMatrix32 &self, UInt col_index) {
      UInt32 idx;
      Real32 min_val;
      self.colMin(col_index, idx, min_val);
      return py::make_tuple(idx, min_val);
    })
    .def("colMax", [](SparseMatrix32 &self, UInt col_index) {
      UInt32 idx;
      Real32 max_val;
      self.colMax(col_index, idx, max_val);
      return py::make_tuple(idx, max_val);
    })
    .def("colSum", &SparseMatrix32::colSum)
    .def("colSums", [](SparseMatrix32 &self) {
      py::array_t<Real32> out(self.nCols());
      self.colSums(arr_begin(out));
      return out;
    })
    .def("colProd", &SparseMatrix32::colProd)
    .def("colProds", [](SparseMatrix32 &self) {
      py::array_t<Real32> out(self.nCols());
      self.colProds(arr_begin(out));
      return out;
    })
    .def("toPyString", [](SparseMatrix32 &self) {
      std::stringstream ss;
      self.toCSR(ss);
      return ss.str();
    })
    .def("fromPyString", [](SparseMatrix32 &self, std::string s) {
      std::stringstream ss(s);
      self.fromCSR(ss);
    })
    .def("clip", &SparseMatrix32::clip)
    .def("add", py::overload_cast<const Real32&>(&SparseMatrix32::add))
    .def("add", py::overload_cast<const SparseMatrix32&>(&SparseMatrix32::add))
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
    .def("addRow", [](SparseMatrix32 &self, py::array_t<Real32> row) {
      self.addRow(arr_begin(row));
    })
    .def("addCol", [](SparseMatrix32 &self, py::array_t<Real32> col) {
      self.addCol(arr_begin(col));
    })
    .def("deleteRows", [](SparseMatrix32 &self, py::array_t<UInt32> rowIndices) {
      self.deleteRows(arr_begin(rowIndices), arr_end(rowIndices));
    })
    .def("deleteCols", [](SparseMatrix32 &self, py::array_t<UInt32> colIndices) {
      self.deleteCols(arr_begin(colIndices), arr_end(colIndices));
    })
    .def("getCol", [](SparseMatrix32 &self, UInt32 col) {
      py::array_t<Int32> out(self.nRows());
      self.getColToDense(col, arr_begin(out));
      return out;
    })
    .def("getOuter", [](SparseMatrix32 &self,
                        py::array_t<UInt32> i,
                        py::array_t<UInt32> j) {
      SparseMatrix32 out(i.size(), j.size());
      self.getOuter(arr_begin(i), arr_end(i), arr_begin(j), arr_end(j), out);
      return out;
    })
    .def("setOuter", [](SparseMatrix32 &self,
                        py::array_t<UInt32> i,
                        py::array_t<UInt32> j,
                        py::array_t<Real32> v) {
      PyArraySparseMatrixMimic v_matrix(v);
      self.setOuter(arr_begin(i), arr_end(i), arr_begin(j), arr_end(j), v_matrix);
    })
    .def("setOuter", [](SparseMatrix32 &self,
                        py::array_t<UInt32> i,
                        py::array_t<UInt32> j,
                        const SparseMatrix32& v) {
      self.setOuter(arr_begin(i), arr_end(i), arr_begin(j), arr_end(j), v);
    })
    .def("getElements", [](SparseMatrix32 &self,
                           py::array_t<UInt32> i,
                           py::array_t<UInt32> j) {
      py::array_t<Real32> out(i.size());
      self.getElements(arr_begin(i), arr_end(i), arr_begin(j), arr_begin(out));
      return out;
    })
    .def("setElements", [](SparseMatrix32 &self,
                           py::array_t<UInt32> i,
                           py::array_t<UInt32> j,
                           py::array_t<Real32> v) {
      self.setElements(arr_begin(i), arr_end(i), arr_begin(j), arr_begin(v));
    })
    .def("getSlice", [](SparseMatrix32 &self,
                        UInt32 i_begin, UInt32 i_end,
                        UInt32 j_begin, UInt32 j_end) {
      SparseMatrix32 out(i_end - i_begin, j_end - j_begin);
      self.getSlice(i_begin, i_end, j_begin, j_end, out);
      return out;
    })
    .def("setSlice", [](SparseMatrix32 &self,
                        UInt32 i_begin,
                        UInt32 j_begin,
                        py::array_t<Real32> v) {
      PyArraySparseMatrixMimic v_matrix(v);
      self.setSlice(i_begin, j_begin, v_matrix);
    })
    .def("setSlice", [](SparseMatrix32 &self,
                        UInt32 i_begin,
                        UInt32 j_begin,
                        const SparseMatrix32& v) {
      self.setSlice(i_begin, j_begin, v);
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
    .def("_initFromCapnpPyBytes", [](SparseMatrix32 &self, py::bytes bytes) {
      nupic::PyCapnpHelper::initFromPyBytes(self, bytes.ptr());
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
    .def("nNonZerosPerCol", [](SparseMatrix32 &self) {
      py::array_t<UInt32> out(self.nCols());
      self.nNonZerosPerCol(arr_begin(out));
      return out;
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
    })
    .def("rightVecProd", [](SparseMatrix32 &self, py::array_t<Real32> x) {
      py::array_t<Real32> out(self.nRows());
      self.rightVecProd(arr_begin(x), arr_begin(out));
      return out;
    })
    .def("leftVecProd", [](SparseMatrix32 &self, py::array_t<Real32> x) {
      py::array_t<Real32> out(self.nCols());
      self.leftVecProd(arr_begin(x), arr_begin(out));
      return out;
    })
    .def("rightVecProdAtNZ", [](SparseMatrix32 &self, py::array_t<Real32> x) {
      py::array_t<Real32> out(self.nRows());
      self.rightVecProdAtNZ(arr_begin(x), arr_begin(out));
      return out;
    })
    .def("rowVecProd", [](SparseMatrix32 &self, py::array_t<Real32> x) {
      py::array_t<Real32> out(self.nRows());
      self.rowVecProd(arr_begin(x), arr_begin(out));
      return out;
    })
    .def("vecMaxAtNZ", [](SparseMatrix32 &self, py::array_t<Real32> x) {
      py::array_t<Real32> out(self.nRows());
      self.vecMaxAtNZ(arr_begin(x), arr_begin(out));
      return out;
    })
    .def("vecMaxProd", [](SparseMatrix32 &self, py::array_t<Real32> x) {
      py::array_t<Real32> out(self.nRows());
      self.vecMaxProd(arr_begin(x), arr_begin(out));
      return out;
    })
    .def("blockRightVecProd", [](SparseMatrix32 &self, UInt32 block_size,
                                 py::array_t<Real32> x) {
      SparseMatrix32 out;
      self.blockRightVecProd(block_size, arr_begin(x), out);
      return out;
    })
    .def("axby", [](SparseMatrix32 &self, UInt32 row, Real32 a, Real32 b,
                    py::array_t<Real32> x) {
      self.axby(row, a, b, arr_begin(x));
    })
    .def("axby", [](SparseMatrix32 &self, Real32 a, Real32 b,
                    py::array_t<Real32> x) {
      self.axby(a, b, arr_begin(x));
    })
    .def("elementNZInverse", &SparseMatrix32::elementNZInverse)
    .def("elementNZLog", &SparseMatrix32::elementNZLog)
    .def("elementSqrt", &SparseMatrix32::elementSqrt)
    .def("abs", &SparseMatrix32::abs)
    .def("negate", &SparseMatrix32::negate)
    .def("transpose", py::overload_cast<>(&SparseMatrix32::transpose))
    .def("transpose", py::overload_cast<SparseMatrix32&>(
           &SparseMatrix32::transpose, py::const_))
    .def("multiply", py::overload_cast<const Real32&>(&SparseMatrix32::multiply))
    .def("multiply", py::overload_cast<const SparseMatrix32&, SparseMatrix32&>(
           &SparseMatrix32::multiply, py::const_))
    .def("subtract", py::overload_cast<const Real32&>(&SparseMatrix32::subtract))
    .def("subtract", py::overload_cast<const SparseMatrix32&>(
           &SparseMatrix32::subtract))
    .def("divide", &SparseMatrix32::divide)
    .def("elementNZMultiply", &SparseMatrix32::elementNZMultiply)
    .def("countWhereEqual", &SparseMatrix32::countWhereEqual)
    .def("whereEqual", [](SparseMatrix32 &self, UInt32 begin_row, UInt32 end_row,
                          UInt32 begin_col, UInt32 end_col, Real32 value) {
      std::vector<UInt32> rows, cols;
      self.whereEqual(begin_row, end_row, begin_col, end_col, value,
                      std::back_inserter(rows), std::back_inserter(cols));
      return py::make_tuple(rows, cols);
    });
}

} // namespace py_sparse_matrix

} // namespace nupic

#endif  // NTA_PY_SPARSE_MATRIX
