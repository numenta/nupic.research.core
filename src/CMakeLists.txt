# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have purchased from
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
project(nupic_core CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF) # toggle for cmake debug


# if( POLICY CMP0046 )
#   cmake_policy(VERSION 3.3)
#   # The new policy for add_dependencies is to now error if a dependency target
#   # is not found. The old policy didn't care and continued through configuration
#   cmake_policy(SET CMP0046 OLD)
# endif()

# if( POLICY CMP0054 )
#   # The OLD behavior for this policy is to dereference variables and interpret
#   # keywords even if they are quoted or bracketed. The NEW behavior is to not
#   # dereference variables or interpret keywords that have been quoted or bracketed.
#   cmake_policy(SET CMP0054 OLD)
# endif()


#
# NuPIC CMake options
#
option(NUPIC_IWYU "Enable include-what-you-use
  (http://include-what-you-use.org/). This requires the iwyu binary to be
  discoverable by CMake's find_program, with a minimum CMake version of 3.3.")
if(${NUPIC_IWYU})
  find_program(iwyu_path NAMES include-what-you-use iwyu)
  if(NOT iwyu_path)
    message(FATAL_ERROR "Could not find the program include-what-you-use")
  endif()
endif()

option(NUPIC_TOGGLE_INSTALL "Toggle whether to install headers, libraries, and executables to the install location for C++ clients" OFF)

#
# Set up compile flags for internal sources
#

set(src_compile_flags "${INTERNAL_CXX_FLAGS_OPTIMIZED}")

if(MINGW)
  # This is for GCC 4.8.x
  # http://stackoverflow.com/questions/10660524/error-building-boost-1-49-0-with-gcc-4-7-0
  set(src_compile_flags "${src_compile_flags} -include cmath")
endif()


#
# Compiler definitions specific to nupic.core code
#

string(TOUPPER ${PLATFORM} platform_uppercase)
string(TOUPPER ${CMAKE_SYSTEM_PROCESSOR} processor_uppercase)

set(src_compiler_definitions
    ${COMMON_COMPILER_DEFINITIONS}
    ${CAPNP_COMPILER_DEFINITIONS}
    -DNTA_OS_${platform_uppercase}
    -DNTA_ARCH_${BITNESS}
    -DNTA_PROCESSOR_${processor_uppercase}
    -DHAVE_CONFIG_H
    -DNTA_INTERNAL
    -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    -DBOOST_NO_WREGEX
    -DNUPIC2
    # -DAPR_DECLARE_STATIC
    # -DAPU_DECLARE_STATIC
    )

if(NOT ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7l" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_ASM)
endif()

if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_ASSERTIONS_ON)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_GNU)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_CLANG)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_MSVC)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MinGW")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_GNU
      -D_hypot=hypot)
endif()


add_definitions(${src_compiler_definitions})


#
# Setup include paths
#
if("${PROJECT_BUILD_ARTIFACTS_DIR}" STREQUAL "")
  set(PROJECT_BUILD_ARTIFACTS_DIR "${PROJECT_BINARY_DIR}/artifacts")
endif()

include_directories(SYSTEM
                    ${PROJECT_BINARY_DIR})

include_directories(${PROJECT_SOURCE_DIR})

#
# Let CMake know where all of the external files are.
#
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${REPOSITORY_DIR}/external/common/
                      ${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/)

message(STATUS "CMAKE_PREFIX_PATH   = ${CMAKE_PREFIX_PATH}")

#
# Extract current git sha and record in nupic/Version.hpp
#
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE NUPIC_CORE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(
  "${PROJECT_SOURCE_DIR}/nupic/Version.hpp.in"
  "${PROJECT_BINARY_DIR}/Version.hpp")

#
# Setup gtest
#
set(src_lib_static_gtest gtest)
add_library(${src_lib_static_gtest}
            STATIC
            ${REPOSITORY_DIR}/external/common/src/gtest/gtest-all.cpp)
set_target_properties(${src_lib_static_gtest}
                      PROPERTIES COMPILE_FLAGS "${src_compile_flags}"
                      LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")


# List all .capnp files here. The C++ files will be generated and included
# when compiling later on.
set(src_capnp_specs_rel
  # nupic/proto/ArrayProto.capnp
  nupic/proto/BitHistory.capnp
  nupic/proto/Cell.capnp
  nupic/proto/Cells4.capnp
  nupic/proto/ClaClassifier.capnp
  nupic/proto/ConnectionsProto.capnp
  # nupic/proto/LinkProto.capnp
  # nupic/proto/Map.capnp
  # nupic/proto/NetworkProto.capnp
  # nupic/proto/PyRegionProto.capnp
  nupic/proto/RandomProto.capnp
  # nupic/proto/RegionProto.capnp
  nupic/proto/Segment.capnp
  nupic/proto/SegmentUpdate.capnp
  nupic/proto/SparseBinaryMatrixProto.capnp
  nupic/proto/SparseMatrixProto.capnp
  nupic/proto/SpatialPoolerProto.capnp
  nupic/proto/SdrClassifier.capnp
  nupic/proto/SvmProto.capnp
  nupic/proto/TemporalMemoryProto.capnp
  # nupic/proto/TestNodeProto.capnp
  # nupic/proto/VectorFileSensorProto.capnp
)

# Create custom command for generating C++ code from .capnp schema files.

foreach(spec ${src_capnp_specs_rel})
  list(APPEND src_capnp_specs ${PROJECT_SOURCE_DIR}/${spec})
  list(APPEND src_capnp_generated_hdrs ${PROJECT_BINARY_DIR}/${spec}.h)
  list(APPEND src_capnp_generated_srcs ${PROJECT_BINARY_DIR}/${spec}.c++)
endforeach(spec)
set(src_capnp_all_spec_hdrs_and_srcs
    ${src_capnp_generated_hdrs}
    ${src_capnp_generated_srcs})

create_capnpc_command("${src_capnp_specs}"
                      ${PROJECT_SOURCE_DIR}
                      ${PROJECT_SOURCE_DIR}
                      ${PROJECT_BINARY_DIR}
                      "${src_capnp_all_spec_hdrs_and_srcs}")


message(STATUS "CAPNP_EXECUTABLE          = ${CAPNP_EXECUTABLE}")
message(STATUS "CAPNP_CMAKE_DEFINITIONS   = ${CAPNP_CMAKE_DEFINITIONS}")
message(STATUS "CAPNP_INCLUDE_DIRS        = ${CAPNP_INCLUDE_DIRS}")
message(STATUS "CAPNP_STATIC_LIB_TARGET   = ${CAPNP_STATIC_LIB_TARGET}")

#
# Common system libraries for shared libraries and executables
#
set(src_common_os_libs)

if("${PLATFORM}" STREQUAL "linux")
  list(APPEND src_common_os_libs pthread dl)
elseif("${PLATFORM}" STREQUAL "darwin")
  list(APPEND src_common_os_libs c++abi)
elseif(MSYS OR MINGW)
  list(APPEND src_common_os_libs psapi ws2_32 wsock32 rpcrt4)
elseif("${PLATFORM}" STREQUAL "windows")
  list(APPEND src_common_os_libs oldnames.lib psapi.lib ws2_32.lib)
endif()

message(STATUS "src_common_os_libs        = ${src_common_os_libs}")


#
# Setup nupic_core_solo static library, consisting of our own sources;
#
# this is an intermediate archive that will be merged with external static
# libraries in a subsequent step to produce the nupic_core "combined" static
# library.
#
set(src_lib_static_nupiccore_solo nupic_core_solo)

set(src_nupiccore_srcs
    nupic/algorithms/Anomaly.cpp
    nupic/algorithms/BitHistory.cpp
    nupic/algorithms/Cell.cpp
    nupic/algorithms/Cells4.cpp
    nupic/algorithms/ClassifierResult.cpp
    nupic/algorithms/CondProbTable.cpp
    nupic/algorithms/Connections.cpp
    nupic/algorithms/GaborNode.cpp
    nupic/algorithms/ImageSensorLite.cpp
    nupic/algorithms/InSynapse.cpp
    nupic/algorithms/OutSynapse.cpp
    nupic/algorithms/Segment.cpp
    nupic/algorithms/SegmentUpdate.cpp
    nupic/algorithms/SDRClassifier.cpp
    nupic/algorithms/SpatialPooler.cpp
    nupic/algorithms/TemporalMemory.cpp
    nupic/algorithms/Svm.cpp
    nupic/encoders/ScalarEncoder.cpp

    ## Depends on BundleIO.cpp
    # nupic/encoders/ScalarSensor.cpp
    # nupic/engine/Collections.cpp  # Depends on BundleIO.cpp
    # nupic/engine/Input.cpp
    # nupic/engine/Link.cpp
    # nupic/engine/LinkPolicyFactory.cpp
    # nupic/engine/Network.cpp
    # nupic/engine/NuPIC.cpp
    # nupic/engine/Output.cpp
    # nupic/engine/Region.cpp
    # nupic/engine/RegionImpl.cpp
    # nupic/engine/RegionImplFactory.cpp
    # nupic/engine/RegionIo.cpp
    # nupic/engine/RegionParameters.cpp
    # nupic/engine/Spec.cpp
    # nupic/engine/TestFanIn2LinkPolicy.cpp
    # nupic/engine/TestNode.cpp
    # nupic/engine/UniformLinkPolicy.cpp
    # nupic/engine/YAMLUtils.cpp

    nupic/math/SparseMatrixAlgorithms.cpp
    nupic/math/SparseMatrixConnections.cpp
    nupic/math/StlIo.cpp
    nupic/math/Topology.cpp
    nupic/ntypes/ArrayBase.cpp
    nupic/ntypes/Buffer.cpp
    # nupic/ntypes/BundleIO.cpp  # Need to remove dependency on APR or fix it
    nupic/ntypes/Collection.cpp
    nupic/ntypes/Dimensions.cpp
    nupic/ntypes/MemParser.cpp
    nupic/ntypes/Scalar.cpp
    nupic/ntypes/Value.cpp

    ## Need to remove dependency on APR or fix it
    # nupic/os/Directory.cpp
    # nupic/os/DynamicLibrary.cpp
    # nupic/os/Env.cpp
    # nupic/os/FStream.cpp
    # nupic/os/OS.cpp
    # nupic/os/OSUnix.cpp
    # nupic/os/OSWin.cpp
    # nupic/os/Path.cpp
    # nupic/os/Regex.cpp

    nupic/os/Timer.cpp

    ## Depends on engine, APR
    # nupic/regions/PyRegion.cpp
    # nupic/regions/VectorFile.cpp
    # nupic/regions/VectorFileEffector.cpp
    # nupic/regions/VectorFileSensor.cpp

    nupic/types/BasicType.cpp
    nupic/types/Fraction.cpp
    # nupic/utils/ArrayProtoUtils.cpp  # Depends on engine
    nupic/utils/LoggingException.cpp
    nupic/utils/LogItem.cpp
    nupic/utils/MovingAverage.cpp
    nupic/utils/Random.cpp
    nupic/utils/StringUtils.cpp
    nupic/utils/TRandom.cpp
    # nupic/utils/Watcher.cpp  # Depends on APR
    )

set(src_lib_static_nupiccore_srcs
    ${src_capnp_generated_srcs}
    ${src_nupiccore_srcs})

set(src_lib_static_nupiccore_compile_flags
    "${src_compile_flags}")

message(STATUS "src_compile_flags = ${src_compile_flags}")
message(STATUS "src_lib_static_nupiccore_compile_flags = ${src_lib_static_nupiccore_compile_flags}")

add_library(${src_lib_static_nupiccore_solo} STATIC
            ${src_lib_static_nupiccore_srcs})
# nupic_core_solo sources depend on headers installed by these external projects
add_dependencies(${src_lib_static_nupiccore_solo}
                 ${YAML_CPP_STATIC_LIB_TARGET}
                 ${YAML_STATIC_LIB_TARGET}
                 # ${APR1_STATIC_LIB_TARGET}
                 # ${APRUTIL1_STATIC_LIB_TARGET}
                 ${Z_STATIC_LIB_TARGET})
set_target_properties(${src_lib_static_nupiccore_solo} PROPERTIES COMPILE_FLAGS
                      ${src_lib_static_nupiccore_compile_flags})
if(${NUPIC_IWYU})
  # TODO: Create a target that doesn't include the generated capnp schema files
  # since we don't want to run iwyu on them and iwyu can only be applied to
  # CMake targets (not source files directly).
  set_target_properties(${src_lib_static_nupiccore_solo}
    PROPERTIES CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()


#
# Create the nupic_core "combined" static library by merging nupic_core_solo
# with our external static libraries
#

include(src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES

set(src_lib_static_nupiccore_combined nupic_core)

set(src_external_static_libs
    ${CAPNP_STATIC_LIB_TARGET}
    ${YAML_CPP_STATIC_LIB_TARGET}
    ${YAML_STATIC_LIB_TARGET}
    # ${APR1_STATIC_LIB_TARGET}
    # ${APRUTIL1_STATIC_LIB_TARGET}
    ${Z_STATIC_LIB_TARGET})

set(src_combined_nupiccore_source_archives
    ${src_lib_static_nupiccore_solo}
    ${src_external_static_libs})

# Create a top-level library target for the combined static lib
merge_static_libraries(${src_lib_static_nupiccore_combined}
                       "${src_combined_nupiccore_source_archives}")


#
# Build tests of the nupic_core "combined" static library
#

# Common libs for test executables
set(src_common_test_exe_libs
    ${src_lib_static_nupiccore_combined}
    ${PYTHON_LIBRARIES}
    ${src_common_os_libs})

message(STATUS "src_common_test_exe_libs = ${src_common_test_exe_libs}")


# Disabled until engine (Network API) is re-added to build
# Setup test_cpp_region
#
# set(src_executable_cppregiontest cpp_region_test)
# add_executable(${src_executable_cppregiontest} test/integration/CppRegionTest.cpp)
# target_link_libraries(${src_executable_cppregiontest} ${src_common_test_exe_libs})
# set_target_properties(${src_executable_cppregiontest} PROPERTIES COMPILE_FLAGS ${src_compile_flags})
# set_target_properties(${src_executable_cppregiontest} PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
# add_custom_target(tests_cpp_region
#                   COMMAND ${src_executable_cppregiontest}
#                   DEPENDS ${src_executable_cppregiontest}
#                   COMMENT "Executing test ${src_executable_cppregiontest}"
#                   VERBATIM)

# Disabled until Network API is re-added to build
# Setup test_py_region
#
# set(src_executable_pyregiontest py_region_test)
# add_executable(${src_executable_pyregiontest} test/integration/PyRegionTest.cpp)
# target_link_libraries(${src_executable_pyregiontest} ${src_common_test_exe_libs})
# set_target_properties(${src_executable_pyregiontest}
#                       PROPERTIES COMPILE_FLAGS ${src_compile_flags})
# set_target_properties(${src_executable_pyregiontest}
#                       PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
# add_custom_target(tests_py_region
#                   COMMAND ${src_executable_pyregiontest}
#                   DEPENDS ${src_executable_pyregiontest}
#                   COMMENT "Executing test ${src_executable_pyregiontest}"
#                   VERBATIM)

#
# Setup test_connections_performance
#
set(src_executable_connectionsperformancetest connections_performance_test)
add_executable(${src_executable_connectionsperformancetest}
               test/integration/ConnectionsPerformanceTest.cpp)
target_link_libraries(${src_executable_connectionsperformancetest}
                      ${src_common_test_exe_libs})
set_target_properties(${src_executable_connectionsperformancetest}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_connectionsperformancetest}
                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
add_custom_target(tests_connections_performance
                  COMMAND ${src_executable_connectionsperformancetest}
                  DEPENDS ${src_executable_connectionsperformancetest}
                  COMMENT "Executing test ${src_executable_connectionsperformancetest}"
                  VERBATIM)

# Disabled until Network API is re-added to build
# Setup helloregion example
#
# set(src_executable_helloregion helloregion)
# add_executable(${src_executable_helloregion} examples/regions/HelloRegions.cpp)
# target_link_libraries(${src_executable_helloregion} ${src_common_test_exe_libs})
# set_target_properties(${src_executable_helloregion}
#                       PROPERTIES COMPILE_FLAGS ${src_compile_flags})
# set_target_properties(${src_executable_helloregion}
#                       PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")

#
# Setup prototest example
#
set(src_executable_prototest prototest)
add_executable(${src_executable_prototest} examples/prototest.cpp)
target_link_libraries(${src_executable_prototest} ${src_common_test_exe_libs})
set_target_properties(${src_executable_prototest}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_prototest}
                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")

#
# Setup HelloSP_TP example
#
set(src_executable_hellosptp hello_sp_tp)
add_executable(${src_executable_hellosptp} examples/algorithms/HelloSP_TP.cpp)
target_link_libraries(${src_executable_hellosptp} ${src_common_test_exe_libs})
set_target_properties(${src_executable_hellosptp}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_hellosptp}
                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")


#
# Setup gtests
#
set(src_executable_gtests unit_tests)
add_executable(${src_executable_gtests}
               test/unit/algorithms/AnomalyTest.cpp
               test/unit/algorithms/Cells4Test.cpp
               test/unit/algorithms/CondProbTableTest.cpp
               test/unit/algorithms/ConnectionsTest.cpp
               test/unit/algorithms/NearestNeighborUnitTest.cpp
               test/unit/algorithms/SDRClassifierTest.cpp
               test/unit/algorithms/SegmentTest.cpp
               test/unit/algorithms/SpatialPoolerTest.cpp
               test/unit/algorithms/SvmTest.cpp
               test/unit/algorithms/TemporalMemoryTest.cpp
               test/unit/encoders/ScalarEncoderTest.cpp
               # test/unit/engine/InputTest.cpp
               # test/unit/engine/LinkTest.cpp
               # test/unit/engine/NetworkTest.cpp
               # test/unit/engine/UniformLinkPolicyTest.cpp
               # test/unit/engine/YAMLUtilsTest.cpp
               test/unit/math/DenseTensorUnitTest.cpp
               test/unit/math/DomainUnitTest.cpp
               test/unit/math/IndexUnitTest.cpp
               test/unit/math/MathsTest.cpp
               test/unit/math/SegmentMatrixAdapterTest.cpp
               test/unit/math/SparseBinaryMatrixTest.cpp
               test/unit/math/SparseMatrix01UnitTest.cpp
               test/unit/math/SparseMatrixTest.cpp
               test/unit/math/SparseMatrixUnitTest.cpp
               test/unit/math/SparseTensorUnitTest.cpp
               test/unit/math/TopologyTest.cpp
               test/unit/ntypes/ArrayTest.cpp
               test/unit/ntypes/BufferTest.cpp
               test/unit/ntypes/CollectionTest.cpp
               test/unit/ntypes/DimensionsTest.cpp
               test/unit/ntypes/MemParserTest.cpp
               test/unit/ntypes/MemStreamTest.cpp
               test/unit/ntypes/NodeSetTest.cpp
               test/unit/ntypes/ScalarTest.cpp
               test/unit/ntypes/ValueTest.cpp
               # test/unit/os/DirectoryTest.cpp
               # test/unit/os/EnvTest.cpp
               # test/unit/os/OSTest.cpp
               # test/unit/os/PathTest.cpp
               # test/unit/os/RegexTest.cpp
               test/unit/os/TimerTest.cpp
               test/unit/types/BasicTypeTest.cpp
               test/unit/types/ExceptionTest.cpp
               test/unit/types/FractionTest.cpp
               test/unit/UnitTestMain.cpp
               test/unit/utils/GroupByTest.cpp
               test/unit/utils/MovingAverageTest.cpp
               test/unit/utils/RandomTest.cpp
               # test/unit/utils/WatcherTest.cpp
               )
target_link_libraries(${src_executable_gtests}
                      ${src_lib_static_gtest}
                      ${src_common_test_exe_libs})
set_target_properties(${src_executable_gtests}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags}
                                 LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
add_custom_target(tests_unit
                  COMMAND ${src_executable_gtests}
                  DEPENDS ${src_executable_gtests}
                  COMMENT "Executing test ${src_executable_gtests}"
                  VERBATIM)

#
# tests_all just calls other targets
#
# TODO This doesn't seem to have any effect; it's probably because the DEPENDS
# of add_custom_target must be files, not other high-level targets. If really
# need to run these tests during build, then either the individual
# add_custom_target of the individual test runners should be declared with the
# ALL option, or tests_all target whould be declared without DEPENDS, and
# add_dependencies should be used to set it's dependencies on the custom targets
# of the inidividual test runners.
add_custom_target(tests_all
                  # DEPENDS tests_cpp_region
                  DEPENDS tests_unit
                  COMMENT "Running all tests"
                  VERBATIM)


#
# Install targets into CMAKE_INSTALL_PREFIX
#
install(TARGETS
        ${src_lib_static_nupiccore_combined}
        ${src_lib_static_gtest}
        # ${src_executable_helloregion}
        # ${src_executable_cppregiontest}
        # ${src_executable_pyregiontest}
        ${src_executable_connectionsperformancetest}
        ${src_executable_hellosptp}
        # ${src_executable_prototest}
        ${src_executable_gtests}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Version.hpp is also used by the nupic.bindings release/deployment system
install(FILES ${PROJECT_BINARY_DIR}/Version.hpp
        DESTINATION include/nupic)

if (NUPIC_TOGGLE_INSTALL)
  install(DIRECTORY nupic DESTINATION include
          MESSAGE_NEVER
          FILES_MATCHING PATTERN "*.h*"
          PATTERN "*.hpp.in" EXCLUDE)
  install(DIRECTORY nupic DESTINATION include
          MESSAGE_NEVER
          FILES_MATCHING PATTERN "*.i")
  install(DIRECTORY nupic DESTINATION include
          MESSAGE_NEVER
          FILES_MATCHING PATTERN "*.capnp")
  install(DIRECTORY ${PROJECT_BINARY_DIR}/nupic DESTINATION include
          MESSAGE_NEVER
          FILES_MATCHING PATTERN "*.capnp.h")

  install(DIRECTORY ${REPOSITORY_DIR}/external/common/include/gtest
          MESSAGE_NEVER
          DESTINATION include/gtest
          FILES_MATCHING PATTERN "*.h*")

  install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
          MESSAGE_NEVER
          DESTINATION include)

  install(DIRECTORY nupic DESTINATION include/nupic
          MESSAGE_NEVER
          FILES_MATCHING PATTERN "*.py")

  foreach(directory ${CAPNP_INCLUDE_DIRS})
    install(DIRECTORY "${directory}/capnp"
            MESSAGE_NEVER
            DESTINATION include)
    install(DIRECTORY "${directory}/kj"
            MESSAGE_NEVER
            DESTINATION include)
  endforeach()

  install(DIRECTORY "${CAPNP_BINARIES}"
          MESSAGE_NEVER
          DESTINATION "${CMAKE_INSTALL_PREFIX}"
          USE_SOURCE_PERMISSIONS
          FILES_MATCHING PATTERN "capnp*")

  install(DIRECTORY "${SWIG_DIR}"
          MESSAGE_NEVER
          DESTINATION share/swig)
  install(PROGRAMS "${SWIG_EXECUTABLE}"
          DESTINATION bin)

  # install(DIRECTORY "${APR1_STATIC_LIB_INC_DIR}/"
  #         MESSAGE_NEVER
  #         DESTINATION include)
  # install(DIRECTORY "${APRUTIL1_STATIC_LIB_INC_DIR}/"
  #         MESSAGE_NEVER
  #         DESTINATION include)

endif (NUPIC_TOGGLE_INSTALL)

#
# `make package` results in
# nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}.tar.gz binary release
#

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}")
include(CPack)
