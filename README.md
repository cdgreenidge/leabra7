# leabra7
[![Build Status](https://travis-ci.org/cdgreenidge/leabra7.svg?branch=master)](https://travis-ci.org/cdgreenidge/leabra7) [![codecov](https://codecov.io/gh/cdgreenidge/leabra7/branch/master/graph/badge.svg)](https://codecov.io/gh/cdgreenidge/leabra7)

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
- Python 3. See the [Hitchhiker's Guide to
  Python](http://python-guide-pt-br.readthedocs.io/en/latest/starting/installation/)
  for installation instructions.

*Note:* If you have Python 2 and Python 3 installed side-by-side, use the
commands `python3` and `pip3` instead of `pip` and `python`.

### Installation
First, clone the repository. It can go anywhere, as long as you do not delete
it after installation:

```
$ git clone https://github.com/cdgreenidge/leabra7.git
```

Now, install the python package in development mode:

```
# Standard
$ cd leabra7
$ python setup.py develop

# Anaconda
$ conda install conda-build
$ cd leabra7
$ conda develop .
```

To check that everything is working correctly, you can install `tox`,
leabra7's test runner, and run all the tests:

```
$ pip install tox
$ tox
```

### Roadmap
See the "Projects" tab for more info.

## For developers

### Setting up your environment (Linux/MacOS only)
Running tests with tox works (and is cross-platform), but tox is slow. To
speed up the development process there is a Makefile that runs common tasks.

First, install the prerequisites. I recommend examining `dev_bootstrap.sh` to
see exactly what will happen:

    $ source dev_bootstrap.sh

A virtual environment will be installed in `~/.virtualenvs/leabra7`. After
activating it, you can use the Makefile to run common tasks.

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
