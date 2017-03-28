# Kabukinai

## Prerequisites

### CUDA

You will need to install CUDA before installing `kabukinai`.

The most straight-forward way of doing this is to go to the NVIDIA CUDA website, and follow the instructions for your platform:

http://docs.nvidia.com/cuda/cuda-installation-guide-linux/

### CMAKE

You will also need to install `cmake` before proceeding.

On Debian based linux systems, this can be done by executing at the command line:

```
# sudo apt-get install cmake
```

On RedHat based linux systems, execute (note that on CentOS you may first need to run `yum install epel-release`):

```
# sudo yum install cmake3
```

On OS X, `cmake` can be installed using [Homebrew](http://brew.sh/)

```
# brew install cmake
```

## Building

To build `kabukinai`, go to the directory containing this file and type at the command line:

```
make
```
