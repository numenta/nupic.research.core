# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022-2024, Numenta, Inc.  Unless you have an agreement
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

import glob
import os
import shutil
import sys

import numpy as np

from setuptools import setup, Extension


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


sources = [
    "src/nupic_module.cpp",
]

debug_mode = os.environ.get("DEBUG_C", False)
build_folder = ("../../build/debug" if debug_mode else "../../build/release")
install_folder = os.environ.get("PREFIX", build_folder)

compile_args = ["-std=c++14"]
link_args = []
extra_objects = [f"{install_folder}/lib/libnupic_research_core.a"]

if debug_mode:
    compile_args += ["-O0", "-D NTA_ASSERTIONS_ON"]
else:
    compile_args += ["-g0"]


if sys.platform == "darwin":
    compile_args += ["-std=c++14", "-mmacosx-version-min=10.10"]
    link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.10"]


if __name__ == "__main__":
    # Copy the proto files into the proto Python package.
    destDir = os.path.relpath(os.path.join("nupic", "proto"))
    for protoPath in glob.glob(os.path.relpath(os.path.join(
            "..", "..", "src", "nupic", "proto", "*.capnp"))):
        shutil.copy(protoPath, destDir)

    version_file = os.path.join("..", "..", "VERSION")
    with open(version_file, encoding="utf8") as f:
        version_str = f.read()

    module = Extension(
        "_nupic",
        sources=sources,
        # This ensures the extension will be rebuilt if these files are changed.
        depends=(glob.glob("../../src/**/*.hpp", recursive=True)
                 + glob.glob("../../src/**/*.h", recursive=True)
                 + glob.glob("../../src/**/*.cpp", recursive=True)),
        extra_objects=extra_objects,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=["./src/",
                      "./../../src/",
                      "./../../src/external",
                      "./../../external/common/include",
                      f"{install_folder}/include",
                      get_pybind_include(),
                      get_pybind_include(user=True),
                      np.get_include()]
    )

    setup(name="nupic.research.core",
          version=version_str,
          description="C++ core for nupic.research",
          packages=["nupic.bindings.math"],
          package_data={
              "nupic.proto": ["*.capnp"],
              "nupic.bindings": ["*.so", "*.pyd"],
              "nupic.bindings.tools": ["*.capnp"],
          },
          setup_requires=["pybind11", "numpy"],
          install_requires=["numpy", "pybind11", "pycapnp>=1.1.0"],
          ext_modules=[module])
