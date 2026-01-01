# Co-simulation of Arbor and TVB

## Setup

First, you'll need a working C++ compiler (GCC), CMake, and an installation of
MPI (OpenMPI). Your OS likely provides packages for these.

Fetch this repository and install dependencies, as follows
``` bash
python3 -mvenv env
source env/bin/activate
pip install -r requirements.txt
CMAKE_ARGS="-DPYTHON_EXECUTABLE=`which python3` -DARB_VECTORIZE=ON -DARB_ARCH=native -DARB_WITH_MPI=ON" pip install --no-cache-dir --no-binary arbor --no-build-isolation arbor 
```
If you have an existing installation of Arbor in this environment, e.g. if something went wrong before,
be sure to `pip uninstall arbor` before executing the last line. This can take a few minutes, since Arbor
is built with optimizations from scratch.

Afterwards, this command
``` bash
python3 -c 'import arbor as A; print(A.config())'
```
should print
```py
{'mpi': False, 'mpi4py': False, 'gpu': None, 'vectorize': True, ... more fields elided}
```

Now, you can build the mechanism catalogue:
``` bash
arbor-build-catalogue the mod
```
this should complete and produce `the-catalogue.so`. 
From here, we assume the catalogue was built successfully and the environment is
active. 
``` bash
bash run.bash
```

