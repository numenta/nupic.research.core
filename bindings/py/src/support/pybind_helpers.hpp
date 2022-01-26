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
 * ---------------------------------------------------------------------
 */

#ifndef NTA_PYBIND_HELPERS
#define NTA_PYBIND_HELPERS

#include <pybind11/numpy.h>

template <typename T>
T* arr_begin(pybind11::array_t<T> &arr)
{
  return (T*)arr.request().ptr;
}

template <typename T>
T* arr_end(pybind11::array_t<T> &arr)
{
  return (T*)arr_begin(arr) + arr.size();
}

#endif // NTA_PYBIND_HELPERS
