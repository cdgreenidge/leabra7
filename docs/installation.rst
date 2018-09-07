Installing leabra7
==================

**Prerequisites:**

- Python 3, with the :code:`conda` package manager. See the `Anaconda
  installation
  guide <https://conda.io/docs/user-guide/install/download.html>`_ for
  installation instructions.

**Installation**

Run the following commands to add the necessary conda channels:

.. code-block:: shell

		$ conda config --append channels pytorch
		$ conda config --append channels conda-forge

Now, you can install leabra7 with

.. code-block:: shell

		$ conda install -c cdg4 leabra7

Now, you can head over to the :doc:`tutorial <tutorial>` to get
started.
