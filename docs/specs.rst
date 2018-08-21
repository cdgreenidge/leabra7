Spec Objects
============

All parameters in a **leabra7** simulation are contained in "spec
objects." A layer's parameters are contained in a
:py:class:`LayerSpec` object, a projection's parameters are contained
in a :py:class:`ProjnSpec` object, and so on. This allows you to
easily create many objects with the same bundle of parameters. For
example, to create three layers with the :code:`inhibition_type`
parameter set to *k-winner-take-all inhibition*, you could use the
following code:

.. code-block:: python

		import leabra7 as lb

		net = lb.Net()

		layer_spec = lb.LayerSpec(inhibition_type="kwta")
		net.new_layer(name="layer1", spec=layer_spec)
		net.new_layer(name="layer2", spec=layer_spec)
		net.new_layer(name="layer3", spec=layer_spec)

If you don't provide a value for a parameter in the spec's
constructor, it will be set to its default value. Here, we create a
layer spec object with all parameters set to their default values:

.. code-block:: python

		>>> default_layer_spec = lb.LayerSpec()

Since parameters are stored as class attributes, we can access them
using dot notation:

.. code-block:: python

		>>> default_layer_spec.inhibition_type
		"fffb"

We see that by default, the layer spec's :code:`inhibition_type`
parameter is set to :code:`"fffb"`, which stands for
*feedforward/feedback inhibition*. To override the value, we would
simply provide the desired value for :code:`inhibition_type` as a
keyword argument in the constructor:

.. code-block:: python

		>>> lb.LayerSpec(inhibition_type="fffb").inhibition_type
		"fffb"

If you try to set a parameter that doesn't exist, an exception will be raised

.. code-block:: python

		>>> lb.LayerSpec(nonexistent_param=3)
		ValueError: nonexistent_param is not a valid parameter name for this spec.

Below is a list of all the specs in leabra7, along with their
parameters and default values:

.. py:class:: LayerSpec()
