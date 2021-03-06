# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2017, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.3)
project(htmresearch_core CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

set(CMAKE_VERBOSE_MAKEFILE OFF)

get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR} ABSOLUTE)


#
# Global NuPIC CMake options
#
option(NUPIC_BUILD_PYEXT_MODULES
       "Turn on building of python extension modules for htmresearch core bindings; turn off to build only static htmresearch core lib with full symbol visibility."
       ON)

message(STATUS "NUPIC_BUILD_PYEXT_MODULES = ${NUPIC_BUILD_PYEXT_MODULES}")
message(STATUS "PY_EXTENSIONS_DIR         = ${PY_EXTENSIONS_DIR}")

message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_C_COMPILER      = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER    = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_BUILD_TYPE      = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX  = ${CMAKE_INSTALL_PREFIX}")


# Identify platform name.
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)

# Define a platform suffix, eg ${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}
if (MSYS OR MINGW)
  set(PLATFORM_SUFFIX -gcc)
endif()


# Make sure we can link against any specified paths.
include_directories(SYSTEM ${CMAKE_INCLUDE_PATH})

set(EP_BASE ${CMAKE_BINARY_DIR}/ThirdParty)

# Determine common toolchain settings, compiler and link flags
include(CommonCompilerConfig)

# Set up builds of external dependencies and get their exports.
# (see individual external/*.cmake modules for exported settings and functions)
add_subdirectory(external)
include_directories(SYSTEM ${EXTERNAL_INCLUDE_DIRS})

# Now build htmresearch_core project.
add_subdirectory(src)
