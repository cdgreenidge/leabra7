Connection Weight Distributions
===============================

.. toctree::
   :maxdepth: 2

.. module:: leabra7

When creating a projection, you can specify the distribution from
which the connection weights are drawn in the projection spec, using
the :code:`dist` parameter. Here is an example of adding a projection
to the network :code:`net`, from the layer named :code:`"input"` to
the layer named :code:`"output"`, with all weights set to the scalar
value :code:`0.6`.

.. code-block:: python

		import leabra7 as lb

		net.new_projn(name="input_to_output",
                              pre="input",
			      post="output",
			      spec=lb.ProjnSpec(dist=lb.Scalar(0.6)))

Below is a list of all supported distributions:

.. py:class:: Scalar(value: float)

	      A scalar "distribution" (i.e. a constant).

	      :param value: The value of the scalar.

.. py:class:: Uniform(low: float, high: float)

	      A uniform distribution.

	      :param low: The lower bound of the distribution's interval, inclusive.
	      :param high: The upper bound of the distribution's interval, inclusive.
	      :raises ValueError: If :code:`low > high`.

.. py:class:: Gaussian(mean: float, var: float)

	      A Gaussian distribution.

	      :param mean: The mean of the distribution.
	      :param var: The variance of the distribution.
	      :raises ValueError: If :code:`var` is negative.

.. py:class:: LogNormal(mean: float, var: float)

	      A lognormal distribution. The parameters :code:`mean`
	      and :code:`var` are for the unique Gaussian random
	      variable :math:`X` such that :math:`Y = e^X`, where
	      :math:`Y` is the lognormal random variable.

	      :param mean: The mean of :math:`X`.
	      :param var: The variance of :math:`X`.
	      :raises ValueError: If :code:`var` is negative.

.. py:class:: Exponential(lambd: float)

	      An exponential distribution.

	      :param lambd: The distribution's rate parameter
                            :math:`\lambda`. If :math:`\mu` is the
                            mean, then :math:`\lambda = 1 / \mu`.
