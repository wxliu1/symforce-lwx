# Build from Source

For best results, you should build from the [latest tagged release](https://github.com/symforce-org/symforce/releases/latest). You can also build from `main`, or from another branch, but everything is less guaranteed to work.

SymForce requires Python 3.8 or later. The build is currently tested on Linux and macOS, SymForce on Windows is untested (see [#145](https://github.com/symforce-org/symforce/issues/145)). We strongly suggest creating a virtual python environment.

Install the `gmp` package with one of:

```
apt install libgmp-dev            # Ubuntu
brew install gmp                  # Mac
conda install -c conda-forge gmp  # Conda
```

SymForce contains both C++ and Python code. The C++ code is built using CMake. You can build the package either by calling pip, or by calling CMake directly. If building with `pip`, this will call CMake under the hood, and run the same CMake build for the C++ components.

If you encounter build issues, please file an [issue](https://github.com/symforce-org/symforce/issues).

## Build with pip

If you just want to build and install SymForce without repeatedly modifying the source, the recommended way to do this is with pip. From the symforce directory:

```
pip install .
```

If you're modifying the SymForce Python sources, you can do an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) instead. This will let you modify the Python components of SymForce without reinstalling. If you're going to repeatedly modify the C++ sources, you should instead build with CMake directly as described [below](https://github.com/symforce-org/symforce?tab=readme-ov-file#build-with-cmake). From the symforce directory:

```
pip install -e .
```

You should then [verify your installation](https://github.com/symforce-org/symforce?tab=readme-ov-file#verify-your-installation).

***Note:*** `pip install .` will not install pinned versions of SymForce's dependencies, it'll install any compatible versions. It also won't install all packages required to run all of the SymForce tests and build all of the targets (e.g. building the docs or running the linters). If you want all packages required for that, you should `pip install .[dev]` instead (or one of the other groups of extra requirements in our `setup.py`). If you additionally want pinned versions of our dependencies, which are the exact versions guaranteed by CI to pass all of our tests, you can install them from `pip install -r dev_requirements.txt`.

*Note: Editable installs as root with the system python on Ubuntu (and other Debian derivatives) are broken on `setuptools<64.0.0`. This is a [bug in Debian](https://ffy00.github.io/blog/02-python-debian-and-the-install-locations/), not something in SymForce that we can fix. If this is your situation, either use a virtual environment, upgrade setuptools to a version `>=64.0.0`, or use a different installation method.*

## Build with CMake

If you'll be modifying the C++ parts of SymForce, you should build with CMake directly instead - this method will not install SymForce into your Python environment, so you'll need to add it to your PYTHONPATH separately.

Install python requirements:

```
pip install -r dev_requirements.txt
```


Build SymForce (requires C++14 or later):

```
mkdir build
cd build
cmake ..
make -j $(nproc)
```

You'll then need to add SymForce (along with `gen/python` and `third_party/skymarshal` within symforce and `lcmtypes/python2.7` within the build directory) to your PYTHONPATH in order to use them, for example:
```
export PYTHONPATH="$PYTHONPATH:/path/to/symforce:/path/to/symforce/build/lcmtypes/python2.7"
```

If you want to install SymForce to use its C++ libraries in another CMake project, you can do that with:
```
make install
```
<font color=red>
SymForce does not currently integrate with CMake's `find_package` (see [#209](https://github.com/symforce-org/symforce/issues/209)), so if you do this you currently need to add its libraries as link dependencies in your CMake project manually.
</font>

## Verify your installation

```
>>> import symforce
>>> symforce.get_symbolic_api()
'symengine'
>>> from symforce import cc_sym
```


If you see `'sympy'` here instead of `'symengine'`, or can't import `cc_sym`, your installation is probably broken and you should submit an [issue](https://github.com/symforce-org/symforce/issues).





# Install

Install with pip:

```
pip install symforce
```


Verify the installation in Python:

```
>>> import symforce.symbolic as sf
>>> sf.Rot3()
```

This installs pre-compiled C++ components of SymForce on Linux and Mac using pip wheels, but does not include C++ headers. If you want to compile against C++ SymForce types (like `sym::Optimizer`), you currently need to [build from source](https://github.com/symforce-org/symforce?tab=readme-ov-file#build-from-source).


**附上安装步骤：**
1. 安装gmp用于浮点数加速运算
```
apt install libgmp-dev
```

2. Install python requirements:
```
pip install -r dev_requirements.txt
```

3. 用CMake编译SymForce
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../../symforce_install ..
make -j $(nproc)
make install
```

4. 在.bashrc中配置环境变量PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:/root/SETUP/symforce:/root/SETUP/symforce/build/lcmtypes/python2.7
```

5. 用python验证
```
>>> import symforce.symbolic as sf
>>> sf.Rot3()
```
```
>>> import symforce
>>> symforce.get_symbolic_api()
'symengine'
>>> from symforce import cc_sym
```