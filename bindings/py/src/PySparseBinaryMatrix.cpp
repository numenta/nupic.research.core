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
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <nupic/math/SparseBinaryMatrix.hpp>

#include <nupic_module.hpp>
#include "support/PyCapnp.hpp"
#include "support/pybind_helpers.hpp"

using nupic::Int32;
using nupic::UInt32;
using nupic::UInt64;
using nupic::Real32;

namespace py = pybind11;


void module_add_SparseBinaryMatrix(py::module &m) {

  typedef nupic::SparseBinaryMatrix<UInt32, UInt32> SparseBinaryMatrix32;

  py::class_<SparseBinaryMatrix32>(m, "SparseBinaryMatrix")
    .def(py::init<UInt32>())
    .def(py::init<UInt32, UInt32>())
    .def("nRows", &SparseBinaryMatrix32::nRows)
    // .def("nCols", py::overload_cast<void>(&SparseBinaryMatrix32::nCols))
    .def("nCols", static_cast<UInt32 (SparseBinaryMatrix32::*)() const>(&SparseBinaryMatrix32::nCols))
    .def("nNonZeros", &SparseBinaryMatrix32::nNonZeros)
    .def("nNonZerosOnRow", &SparseBinaryMatrix32::nNonZerosOnRow)
    .def("nNonZerosPerRow", [](SparseBinaryMatrix32 &self) {
      py::array_t<Int32> out(self.nRows());
      self.nNonZerosPerRow(arr_begin(out), arr_end(out));
      return out;
    })
    .def("nNonZerosPerCol", [](SparseBinaryMatrix32 &self) {
      py::array_t<Int32> out(self.nCols());
      self.nNonZerosPerCol(arr_begin(out), arr_end(out));
      return out;
    })
    .def("resize", &SparseBinaryMatrix32::resize)
    .def("get", &SparseBinaryMatrix32::get)
    .def("getVersion", &SparseBinaryMatrix32::getVersion, "",
         py::arg("binary") = false)
    .def("getRow", [](SparseBinaryMatrix32 &self, UInt32 row) {
      py::array_t<Int32> out(self.nCols());
      self.getRow(row, arr_begin(out), arr_end(out));
      return out;
    })
    .def("getCol", [](SparseBinaryMatrix32 &self, UInt32 col) {
      py::array_t<Int32> out(self.nRows());
      self.getColToDense(col, arr_begin(out), arr_end(out));
      return out;
    })
    .def("toDense", [](SparseBinaryMatrix32 &self) {
      py::array_t<bool> out({self.nRows(), self.nCols()});
      self.toDense(arr_begin(out), arr_end(out));
      return out;
    })
    .def("_fromDense", [](SparseBinaryMatrix32 &self, UInt32 nrows, UInt32 ncols,
                          py::array_t<bool> data) {
      self.fromDense(nrows, ncols, arr_begin(data), arr_end(data));
    })
    .def("replaceSparseRow", [](SparseBinaryMatrix32 &self, UInt32 row,
                                py::array_t<Real32> x) {
      self.replaceSparseRow(row, arr_begin(x), arr_end(x));
    })
    .def("appendSparseRow", [](SparseBinaryMatrix32 &self, py::array_t<UInt32> x) {
      self.appendSparseRow(arr_begin(x), arr_end(x));
    })
    .def("appendDenseRow", [](SparseBinaryMatrix32 &self, py::array_t<UInt32> x) {
      self.appendDenseRow(arr_begin(x), arr_end(x));
    })
    .def("_writeAsCapnpPyBytes", [](SparseBinaryMatrix32 &self) {
      return nupic::PyCapnpHelper::writeAsPyBytes(self);
    })
    .def("_initFromCapnpPyBytes", [](SparseBinaryMatrix32 &self, PyObject* pyBytes) {
      nupic::PyCapnpHelper::initFromPyBytes(self, pyBytes);
    })
    .def("_rightVecSumAtNZ", [](SparseBinaryMatrix32 &self,
                                py::array_t<UInt32> x,
                                py::array_t<Int32> out) {
      self.rightVecSumAtNZ(arr_begin(x), arr_end(x), arr_begin(out), arr_end(out));
    });
}
