<img src="http://numenta.org/87b23beb8a4b7dea7d88099bfb28d182.svg" alt="NuPIC Logo" width=100/>

# NuPIC Research Core

This repository contains C++ code for [nupic.research](https://github.com/numenta/nupic.research). See the [nupic.research README](https://github.com/numenta/nupic.research) for more information.

## Building from Source

### Prerequisites

- [CMake](http://www.cmake.org/)

### Developer Installation

This option is for developers that would like the ability to do incremental builds of the C++ or for those that are using the C++ libraries directly.

#### Configure and generate C++ build files:

    mkdir -p build/scripts
    cd build/scripts
    cmake ../../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../release

Notes:

- This will generate Release build files. For a debug build, change `-DCMAKE_BUILD_TYPE` to `Debug`.
- If you have dependencies precompiled but not in standard system locations then you can specify where to find them with `-DCMAKE_PREFIX_PATH` (for bin/lib) and `-DCMAKE_INCLUDE_PATH` (for header files).
- The `-DCMAKE_INSTALL_PREFIX=../release` option shown above is optional, and specifies the location where `nupic.research.core` should be installed. If omitted, `nupic.research.core` will be installed in a system location. Using this option is useful when testing versions of `nupic.research.core` language bindings in [`bindings`](bindings).
- To use Include What You Use during compilation, pass `-DNUPIC-IWYU=ON`. This requires that IWYU is installed and findable by CMake, with a minimum CMake version of 3.3. IWYU can be installed from https://include-what-you-use.org/ for Windows and Linux, and on OS X using https://github.com/jasonmp85/homebrew-iwyu.

#### Build:

    # While still in nupic.research.core/build/scripts
    make -j3

> **Note**: The `-j3` option specifies '3' as the maximum number of parallel jobs/threads that Make will use during the build in order to gain speed. However, you can increase this number depending your CPU.

#### Install:

    # While still in nupic.research.core/build/scripts
    make install

#### Run the tests:

    cd ../release/bin
    ./unit_tests

#### Install nupic.bindings Python library:

    # From nupic.research.core
    cd bindings/py
    pip install -r requirements.txt
    pip install -e .

Once it is installed, you can import NuPIC bindings library to your python script using:

    import nupic.bindings

Any time you change the underlying C++ code, you must repeat the above steps:

    # From nupic.research.core/build/scripts
    make -j3
    make install
    # From nupic.research.core
    pip install -e .

## Adding to the bindings

Suppose you want to add bindings for `src/nupic/algorithms/MyCode.cpp`. Follow these steps:

- Create `/bindings/py/src/PyMyCode.hpp`, following the example of other files in that folder.
  - Implement the function `nupic::py_my_code::add_to`.
- Call this function in `/bindings/py/src/nupic_module.cpp`.
  - Import `PyMyCode.hpp`
  - Call `nupic::py_my_code::add_to`
  - With these additions, your code is now now available to Python via the `_nupic` module.
- Explicitly import your code somewhere in `nupic.bindings`
  - For example, add `bindings/py/nupic/algorithms/my_code.py`, which could contain `from _nupic import my_function, MyClass`
  - Optional: add Python functions or classes to this file. You may want to subclass `MyClass` and add additional Python logic.

Whenever you change `src/nupic/algorithms/MyCode.cpp`, remember to rebuild (`make -j3` from `build/scripts`, as above), then run `pip install -e .` from the folder `bindings/py` to rebuild the bindings.

### Using graphical interface

#### Generate the IDE solution:

 * Open CMake executable.
 * Specify the source folder (`$NUPIC_CORE/src`).
 * Specify the build system folder (`$NUPIC_CORE/build/scripts`), i.e. where IDE solution will be created.
 * Click `Generate`.
 * Choose the IDE that interest you (remember that IDE choice is limited to your OS, i.e. Visual Studio is available only on CMake for Windows).

#### Build:

 * Open `nupic_research_core.*proj` solution file generated on `$NUPIC_CORE/build/scripts`.
 * Run `ALL_BUILD` project from your IDE.

#### Run the tests:

 * Run any `tests_*` project from your IDE (check `output` panel to see the results).

## Build Documentation

Run doxygen, optionally specifying the version as an environment variable:

    PROJECT_VERSION=`cat VERSION` doxygen

The results will be written out to the `html` directory.
