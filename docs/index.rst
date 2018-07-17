.. leabra7 documentation master file, created by
   sphinx-quickstart on Mon Jul  2 11:50:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to leabra7's documentation!
===================================

leabra7 is an implementation of the
"Local, Error-driven and Associative, Biologically Realistic Algorithm"
(`LEABRA <https://grey.colorado.edu/emergent/index.php/Leabra>`_)
in Python. It targets quantitative equivalence with the long-term
support emergent71 branch of the
`Emergent project <https://grey.colorado.edu/emergent/index.php/Main_Page>`_
(note: this is not the current version of emergent).

Guide
=====

* :ref:`Purpose`
* :ref:`Brief Overview <brief_overview>`
* :ref:`User Installation Guide <install_guide>`
* :ref:`Documentation`
* :ref:`Developer's Guide <develop_guide>`
* :ref:`Contributors`

.. _Purpose:

Purpose
-------

`Emergent <https://grey.colorado.edu/emergent/index.php/Main_Page>`_ is a
powerful framework for computational neuroscience simulation, but the project
is not readily mod-able and adaptable for new models and new learning algorithms.
This python library seeks to offer an adaptive framework inspired by
`Randy O'Reilly's <http://psych.colorado.edu/~oreilly>`_
work. We hope this library will be indefinitely tweaked and adjusted to create
a myriad of models to inform the study of computational neuroscience. We hope
that this framework serves as a platform for inter-institutional and international
collaboration and code sharing.

.. _brief_overview:

Brief Overview
--------------

Networks are composed of layers and projections. Layers represent groups of units.
These units are not themselves interconnected but are mutually inhibitory. These
layers can be connected to each other through projections. These projections "project"
activations forward as input to the next layer. These values are fed through a weight matrix.
Projections can also have special setups, namely they can be made sparse, such that
there are some percent fewer connections than full connectivity (every unit in the pre layer
connected to every unit in the post layer). Projections can also involve "masking" either the prelayer
or postlayer or both, so that only a few units of each layer are connected. The weights of these
projections is updated according to the LEABRA algorithm.

Networks execute events. These events can be assembled into a sequence of instructions
to train and/or test a network. Events include cycling the activations of the layers and
clamping the activations of layers. See our sample notebooks to see how events
can be used to train and test a network.


.. _install_guide:

User Installation Guide
-----------------------

Prerequisites:
^^^^^^^^^^^^^^

* Anaconda Distribution of Python 3. See the
  `Anaconda Installation Guide. <https://conda.io/docs/user-guide/install/download.html>`_
* The `conda package manager. <https://www.anaconda.com/distribution>`_

Installation:
^^^^^^^^^^^^^

Run the following commands to add the necessary conda channels:

.. code::

  $ conda config --append channels pytorch
  $ conda config --append channels conda-forge

Now you can install leabra7 with:

.. code::

  $ conda install -c cdg4 leabra7


.. note::
  If you have Python 2 and Python 3 installed side-by-side, use the
  commands ``python3`` and ``pip3`` instead of ``pip`` and ``python``.

.. _Documentation:

Documentation
--------------

.. toctree::
   :maxdepth: 4

   leabra7

.. _develop_guide:

Developer's Guide
-----------------

`Github Repo <https://github.com/cdgreenidge/leabra7>`_

`Gitter Chat <https://gitter.im/leabra7/Lobby>`_

.. _Contributors:

Contributors
-------------

This project was spearheaded by Daniel Greenidge with assistance from Noam Miller
and Fabien Benureau.

This a project of `Norman Lab. <https://compmem.princeton.edu>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
