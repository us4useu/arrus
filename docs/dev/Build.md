# Building

To obtain the source files:
```
git clone https://github.com/us4useu/arrus.git
```

## Binaries

### Prerequisites

Note: on Linux you can use environment created with the `.docker/build/Dockerfile`.
The docker image contains all the packages needed to build ARRUS packages
and documentation.
To use it, change current directory to this repositorym, then call:
```
docker build -f .docker/build/Dockerfile --build-arg TARGETPLATFORM=linux/amd64 --no-cache -t us4useu/arrus-build
docker run -it --rm us4useu/arrus-build:latest
```
For `linux_aarch64` remember to change `TARGETPLATFORM` to `linux/arm64`.

All platforms and languages:
- [CMake](https://cmake.org) 3.17 at least
- [Python](https://python.org) 3.8 at least ([conda](https://docs.conda.io/en/latest/miniconda.html) is recommended)
- the following Python packages: `pip install conan pydevops-us4us==0.1.0`
- us4r drivers API headers and binaries (ask us4us developers for access)

`windows_x86_64`:
- the latest Windows 10
- MSVC 2017

`linux_x86_64` and `linux_aarch64`:
- Ubuntu 20.04 at least
- gcc >= 9.0,
- patchelf: `apt-get install patchelf`
- make sure that you are using libstdc++ for new ABI in your default conan profile
  `~/.conan/profiles/default`, the field `compiler.libcxx` should be set to `libstdc++11`

### Building C++ API (*core*)

The simplest option is to use `pydevops`. Change your current directory
to this repository, then select the parameters to be used during the build,
for example:

```
pydevops --clean --options build_type=Debug j=8 us4r_api_dir=/path/to/us4us/libraries
```
The above call will run `conan` and `cmake` configuration steps, build
the sources using OS-dependent generator (MSBuild or Unix Makefiles),
then install the result to the `./install` directory. Please check
`devops.py` file for more information what defaults are available.

Then, if you make any change to the source code and want to rebuild only
the last change, just call:
```
pydevops
```

#### Build Settings

The following `pydevops` and CMake options are available:
- `us4r_api_dir`: path to the us4R API headers and binaries,
- `build_type`: set one common build type for conan, cmake and unit tests
- `py` (optional, default: `OFF`): turns on Python API whl build (alias for CMake's `ARRUS_BUILD_PY`)
- `matlab` (optional, default: `OFF`): turns on MATLAB Toolbox build (alias for CMake's `ARRUS_BUILD_MATLAB`)
- `docs` (optional, default: `OFF`): turns on documentation build (alias for CMake's `ARRUS_BUILD_DOCS`)
- `tests` (optional, default: `OFF`): turns on unit tests build (alias for CMake's `ARRUS_RUN_TESTS`)
- `j`: specify number of jobs that should run the build

The following CMake options are available:
- `ARRUS_EMBED_DEPS` (optional, default: `OFF`): `ON` means that all the dynamically 
   linked binary dependencies will be copied to the `lib64` at the `install` stage
- `ARRUS_APPEND_VERSION_SUFFIX_DATE` (optional, default: `OFF`): `ON` means that 
   to the project's name a suffix with the current date will be appended

#### Python

##### Prerequisites
- the following Python packages: `setuptools virtualenv wheel`
- [SWIG](http://www.swig.org/download.html) at least 4.0.1

##### Building

Just remember to use option `py=on`

```
pydevops --clean --options j=8 py=on us4r_api_dir=/path/to/us4us/libraries
```

#### MATLAB

##### Prerequisites

##### Building

Just remember to use option `matlab=on`

```
pydevops --clean --options j=8 matlab=on us4r_api_dir=/path/to/us4us/libraries
```

#### Documentation

##### Prerequisites

All platforms:
- latest [Doxygen](https://doxygen.nl/) package,
- the following Python packages:
```
pip install sphinx==3.3.1 sphinx_rtd_theme==0.5.0 six breathe docutils==0.16 Jinja2==3.0.3 
pip install "git+https://github.com/pjarosik/matlabdomain@master#egg=sphinxcontrib-matlabdomain
```

`windows_x86_64`

- The latest [MiKTeX](https://miktex.org/) with packages: `ha-prosper`, `prosper`, `latex-tools`
- install the latest version of [strawberry perl](strawberryperl.com)

`linux_x86_64` and `Linux aarch64`:
```
sudo apt-get install texlive-base texlive-fonts-recommended texlive-latex-recommended texlive-latex-extra
sudo apt-get install latexmk
```

##### Building

Just remember to use option `docs=on`

```
pydevops --clean --options docs=on
```

##### Installing (python)
After the build is completed, the package is in `build/api/python`.
To install use `pip install -e build/api/python` command.
