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

#ifndef NTA_PYBIND_MATH
#define NTA_PYBIND_MATH

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <vector>

#include "nupic/math/SparseBinaryMatrix.hpp"
#include "nupic/math/SparseMatrix.hpp"
#include "nupic/math/SparseMatrixConnections.hpp"
#include "nupic/utils/Random.hpp"
#include "support/PyCapnp.hpp"
#include "support/pybind_helpers.hpp"

using std::pair;
using std::vector;
using nupic::SparseMatrixConnections;
using nupic::Int32;
using nupic::UInt32;
using nupic::UInt64;
using nupic::Real32;
using nupic::Random;

namespace py = pybind11;


void module_add_math(py::module &m) {
  py::class_<Random>(m, "Random")
    .def(py::init<>())
    .def(py::init<UInt64>())
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::pickle(
           [](const Random &self) { // __getstate__
             std::stringstream ss;
             ss << self;
             return ss.str();
           },
           [](const std::string &s) { // __setstate__
             Random self;
             std::stringstream ss(s);
             ss >> self;
             return self;
           }
           )
      )
    .def("getUInt32", &Random::getUInt32, "", py::arg("max") = Random::MAX32)
    .def("getUInt64", &Random::getUInt64, "", py::arg("max") = Random::MAX64)
    .def("getReal64", &Random::getReal64)
    .def("shuffle", [](Random &self, py::array arr) {
      if (arr.ndim() != 1) {
        throw std::invalid_argument("Only one dimensional arrays are supported.");
      }

      if (arr.itemsize() == 4) {
        auto a = arr.mutable_unchecked<UInt32, 1>();
        self.shuffle((UInt32*)a.data(0), (UInt32*)a.data(0) + a.size());
        return;
      } else if (arr.itemsize() == 8) {
        auto a = arr.mutable_unchecked<UInt64, 1>();
        self.shuffle((UInt64*)a.data(0), (UInt64*)a.data(0) + a.size());
        return;
      } else {
        throw std::invalid_argument("Unsupported data size. Expected 32 or 64-bit.");
      }
    })
    .def("sample", [](Random &self, py::array population, py::array choices) {
      if (population.ndim() != 1 || choices.ndim() != 1) {
        throw std::invalid_argument("Only one dimensional arrays are supported.");
      }
      if (population.itemsize() != choices.itemsize()) {
        throw std::invalid_argument(
          "Type of value in population and choices arrays must match.");
      }
      if (choices.size() > population.size()) {
        throw std::invalid_argument(
          "Population size must be greater than number of choices");
      }

      switch (population.itemsize()) {
        case 4:
          self.sample((UInt32*)population.mutable_unchecked<UInt32, 1>().data(0),
                      population.size(),
                      (UInt32*)choices.mutable_unchecked<UInt32, 1>().data(0),
                      choices.size());
          break;
        case 8:
          self.sample((UInt64*)population.mutable_unchecked<UInt64, 1>().data(0),
                      population.size(),
                      (UInt64*)choices.mutable_unchecked<UInt64, 1>().data(0),
                      choices.size());
          break;
        default:
          throw pybind11::type_error(
            "Unsupported data size. Expected 32 or 64-bit.");
      }
    })
    .def("initializeReal32Array", [](Random &self, py::array_t<Real32> array) {
      for (auto it = arr_begin(array); it != arr_end(array); ++it) {
        *it = self.getReal64();
      }
    });

  typedef nupic::SparseMatrix<UInt32, Real32, Int32, Real32, nupic::DistanceToZero<Real32>> SparseMatrix32;

  py::class_<SparseMatrix32>(m, "SparseMatrix")
    .def(py::init<>())
    .def(py::init<UInt32, UInt32>())
    .def("nRows", &SparseMatrix32::nRows)
    .def("nCols", &SparseMatrix32::nCols)
    .def("resize", &SparseMatrix32::resize)
    .def("get", &SparseMatrix32::get)
    .def("set", &SparseMatrix32::set)
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
}

#endif // NTA_PYBIND_MATH
