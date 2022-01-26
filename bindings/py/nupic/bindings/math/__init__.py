# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np

import _nupic

from .nupic_random import Random
from .sparse_binary_matrix import SparseBinaryMatrix
from .sparse_matrix import SparseMatrix
from .sparse_matrix_connections import SparseMatrixConnections


def GetNTAReal():
    return np.float32


# Included for compatibility with code written for nupic.core
SM32 = SparseMatrix
SM_01_32_32 = SparseBinaryMatrix

__exports__ = [
    "GetNTAReal",
    "Random",
    "SM32",
    "SM_01_32_32",
    "SparseBinaryMatrix",
    "SparseMatrix",
    "SparseMatrixConnections",
]
