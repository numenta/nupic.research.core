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

#ifndef NTA_PY_RANDOM
#define NTA_PY_RANDOM

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <vector>

#include <nupic/utils/Random.hpp>

#include <nupic_module.hpp>
#include "support/PyCapnp.hpp"
#include "support/pybind_helpers.hpp"

namespace nupic {
namespace py_random {

using std::pair;
using std::vector;
using nupic::Int32;
using nupic::UInt32;
using nupic::UInt64;
using nupic::Real32;
using nupic::Random;

namespace py = pybind11;


void add_to(py::module &m) {
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
    .def("_writeAsCapnpPyBytes", [](Random &self) {
      return nupic::PyCapnpHelper::writeAsPyBytes(self);
    })
    .def("_initFromCapnpPyBytes", [](Random &self, py::object pyBytes) {
      nupic::PyCapnpHelper::initFromPyBytes(self, pyBytes.ptr());
    })
    .def("getUInt32", &Random::getUInt32, "", py::arg("max") = Random::MAX32)
    .def("getUInt64", &Random::getUInt64, "", py::arg("max") = Random::MAX64)
    .def("getReal64", &Random::getReal64)
    .def("getSeed", &Random::getSeed)
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
}

} // namespace py_random
} // namespace nupic

#endif // NTA_PY_RANDOM
