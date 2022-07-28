set -x 
declare -a CMAKE_PLATFORM_FLAGS
if [[ ${HOST} =~ .*darwin.* ]]; then
    CMAKE_PLATFORM_FLAGS+=(-DCMAKE_OSX_SYSROOT="${CONDA_BUILD_SYSROOT}")
fi

if [[ ${DEBUG_C} == yes ]]; then
  CMAKE_BUILD_TYPE=Debug
else
  CMAKE_BUILD_TYPE=Release
fi

# Build C++ library
cmake ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DCMAKE_INSTALL_LIBDIR="lib" \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DCMAKE_C_FLAGS_RELEASE="${CFLAGS}" \
      -DCMAKE_C_FLAGS_DEBUG="${CFLAGS}" \
      ${CMAKE_PLATFORM_FLAGS[@]} \
      ${EXTRA_CMAKE_ARGS} \
      ${SRC_DIR}
make -j${CPU_COUNT} ${VERBOSE_CM}
make install -j${CPU_COUNT}

# Build python bindings
pushd ${SRC_DIR}/bindings/py
${PYTHON} -m pip install .
