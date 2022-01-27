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

#include <vector>

#include "nupic_module.hpp"
#include "PyApicalTiebreakTemporalMemory.hpp"
#include "PyConnections.hpp"
#include "PyRandom.hpp"
#include "PySparseBinaryMatrix.hpp"
#include "PySparseMatrixConnections.hpp"
#include "PySparseMatrix.hpp"

namespace py = pybind11;


PYBIND11_MODULE(_nupic, m)
{
  nupic::py_connections::add_to(m);
  nupic::py_apical_tiebreak_temporal_memory::add_to(m);
  nupic::py_random::add_to(m);
  nupic::py_sparse_matrix::add_to(m);
  nupic::py_sparse_binary_matrix::add_to(m);
  nupic::py_sparse_matrix_connections::add_to(m);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
