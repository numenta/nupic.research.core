/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2024, Numenta, Inc.  Unless you have an agreement
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

#ifndef NTA_PY_SPATIAL_POOLER
#define NTA_PY_SPATIAL_POOLER

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <vector>

#include <nupic/algorithms/SpatialPooler.hpp>

#include <nupic_module.hpp>
#include "support/PyCapnp.hpp"
#include "support/pybind_helpers.hpp"

namespace nupic {
namespace py_spatial_pooler {

using namespace nupic::algorithms::spatial_pooler;

namespace py = pybind11;


void add_to(py::module &m) {
  py::class_<SpatialPooler>(m, "SpatialPooler")
    .def(py::init<>());
}

} // namespace py_spatial_pooler
} // namespace nupic

#endif // NTA_PY_SPATIAL_POOLER
