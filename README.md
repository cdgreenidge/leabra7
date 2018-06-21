# leabra7
[![Anaconda-Server Badge](https://anaconda.org/cdg4/leabra7/badges/version.svg)](https://anaconda.org/cdg4/leabra7) [![Anaconda-Server Badge](https://anaconda.org/cdg4/leabra7/badges/platforms.svg)](https://anaconda.org/cdg4/leabra7) [![Anaconda-Server Badge](https://anaconda.org/cdg4/leabra7/badges/latest_release_date.svg)](https://anaconda.org/cdg4/leabra7)

| linux | windows | coverage |
|-------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [![Build Status](https://travis-ci.org/cdgreenidge/leabra7.svg?branch=windows-ci)](https://travis-ci.org/cdgreenidge/leabra7) | [![Build status](https://ci.appveyor.com/api/projects/status/pu47got47lql75j2/branch/master?svg=true)](https://ci.appveyor.com/project/cdgreenidge/leabra7/branch/master) | [![codecov](https://codecov.io/gh/cdgreenidge/leabra7/branch/master/graph/badge.svg)](https://codecov.io/gh/cdgreenidge/leabra7) |

**leabra7** is an implementation of the "Local, Error-driven and Associative,
Biologically Realistic Algorithm" in Python. It targets quantitative
equivalence with the long-term support `emergent71` branch of the [Emergent
project](https://grey.colorado.edu/emergent/index.php/Main_Page).

Why is this interesting? Current neural network technology struggles with
recurrence and focuses on global learning algorithms. The leabra algorithm
allows simulation of neural networks with massive recurrence and local
learning algorithms. Currently, we are using it explore interaction
between the hippocampus and neocortex during memory recall (see the
[Princeton Computational Memory Lab](https://compmem.princeton.edu/) for more
details).

To see it in action, look at the IPython notebook in `docs`. It's not very
verbose but the docs will be fully written once the API stabilizes (0.1 release
targeted in roughly 1 month).

## Getting started

### Prerequisites
- Anaconda Distribution of Python 3. See the [Anaconda Installation Guide](https://conda.io/docs/user-guide/install/download.html)
  for installation instructions.
- The [conda](https://www.anaconda.com/distribution/) package manager.

*Note:* If you have Python 2 and Python 3 installed side-by-side, use the
commands `python3` and `pip3` instead of `pip` and `python`.

### Installation

### For users

Run the following commands to add the necessary conda channels:

```
$ conda config --append channels pytorch
$ conda config --append channels conda-forge
```

Now, you can install leabra7 with

```
$ conda install -c cdg4 leabra7
```

### For developers

**Gitter chat is [here](https://gitter.im/leabra7/Lobby).**

First, clone the repository. It can go anywhere, as long as you do not delete
it after installation:

```
$ git clone https://github.com/cdgreenidge/leabra7.git
```

Run the following commands to add the necessary conda channels and
create a virtual environment for development:

```
$ conda config --append channels pytorch
$ conda config --append channels conda-forge
$ conda env create -f scripts/environment.yml
```

This will create a new conda environment, named `leabra7`, and install the
dependencies necessary for package development. Once it is created, activate it
with

```
$ source activate leabra7
```

Install the leabra7 package in development mode:

```
$ conda-develop .
```

Now, run static analysis and tests to check that everything is working:

```
$ make
```

At this point, you can use `leabra7` like a normal Python
package. Changes made to the files will be reflected in the Python
interpreter, as long as the package is reloaded or the interpreter is
restarted.

### Roadmap
See the "Projects" tab for more info.

## For developers

### Style
* "I hate code, and I want as little of it as possible in our product."
  – Jack Diederich
* In general, follow the [Khan Academy style
  guide](https://github.com/Khan/style-guides/blob/master/style/python.md).
* Don't commit code that produces broken tests, or code
  that produces warnings during the build process. Disable warnings only if
  absolutely necessary. Think three times about committing untested code (in
  general, this should only be core simulation code that doesn't have clear
  outputs or properties.)
* Read the [Suckless philosophy](http://suckless.org/philosophy) and the
  [Unix philosophy](http://www.faqs.org/docs/artu/ch01s06.html) for
  inspiration.

## Contributors
Special thanks to Fabien Benureau for providing parts of the NXX1
implementation.
