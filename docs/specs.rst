Spec Objects
============

.. toctree::
   :maxdepth: 2

.. module:: leabra7

.. contents:: :local:

Intro to Spec objects
---------------------

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

		>>> lb.LayerSpec(inhibition_type="kwta").inhibition_type
		"kwta"

If you try to set a parameter that doesn't exist, an exception will be raised

.. code-block:: python

		>>> lb.LayerSpec(nonexistent_param=3)
		ValueError: nonexistent_param is not a valid parameter name for this spec.

Below is a list of all the specs in leabra7, along with their
parameters and default values:

LayerSpec
---------

.. py:class:: LayerSpec()

   Contains parameters for network layers.

   .. py:attribute:: inhibition_type

      The type of inhibition. Can be :code:`"fffb"` for standard
      feedforward/feedback inhibition, :code:`"kwta"` for
      k-winner-take-all inhibition, or :code:`"kwta_avg"` for
      average-based k-winner-take-all inhibition. Defaults to
      :code:`"fffb"`.

      KWTA inhibition sets the inhibition so that only *k* units have
      a :code:`"v_m_eq"` value above their firing threshold.

      KWTA-average inhibition is slightly more complicated:
      for each unit in a layer of size :math:`n`, it calculates the
      inhibition :math:`gi_{thr}` that will put it at its firing
      threshold. Then it calculates two averages:

      1. The average of the top :math:`k` values of :math:`gi_{thr}`,
         called :code:`gi_k`.
      2. The average of the bottom :math:`n - k` values of
         :math:`gi_{thr}`, called :code:`gi_nk`

      The final inhibition is a convex combination of these two
      values: :code:`p * gi_k + (1 - p) * gi_nk`. The value of
      :code:`p` is determined by :attr:`kwta_pt`.

   .. py:attribute:: kwta_pct

      The proportion of winners for KWTA inhibition. Defaults to
      :code:`0.1`.

   .. py:attribute:: kwta_pt

      In KWTA-average inhibition, the convex combination parameter
      :code:`p`. See :attr:`inhibition_type`. Defaults to
      :code:`0.5`. Valid values are any float in the range :math:`[0,
      1]`.

   .. py:attribute:: ff

      The feedforward inhibition multiplier for feedforward/feedback
      inhibition. Controls the relative strength of feedforward
      inhibition. Defaults to :code:`1.0`. Valid values are any float
      in the range :math:`[0, \infty)`.

   .. py:attribute:: ff0

      The feedforward inhibition offset. A layer with average
      activation below this level will not trigger feedforward
      inhibition. Defaults to :code:`0.1`. Valid values are any float
      in the range :math:`[0, \infty]`.

   .. py:attribute:: fb

      The feedback inhibition multiplier for feedforward/feedback
      inhibition. Controls the relative strength of feedback
      inhibition. Defaults to :code:`1.0`. Valid values are any float
      in the range :math:`[0, \infty)`.

   .. py:attribute:: fb_dt

      The feedback inhibition integration time constant. Defaults to
      :code:`1 / 1.4`. Valid values are any float in the range
      :math:`[0, \infty]`.

   .. py:attribute:: gi

      The global inhibition multiplier. Controls the relative strength
      of total (feedforward and feedback) inhibition. Defaults to
      :code:`1.8`. Valid values are any float in the range :math:`[0,
      \infty]`.

   .. py:attribute:: avg_dt

      The integration constant for the :code:`cos_diff_avg` error
      metric (between plus and minus phase activations), which can be
      used to modulate the proportion of Hebbian learning and the
      learning rate. Defaults to :code:`0.01`. Valid values are any
      float in the range :math:`[0, \infty]`.

   .. py:attribute:: clamp_max

      Typically, units in input and output layers are clamped to
      binary values (0 or 1). But units cannot support an activation
      of 1. Thus any value above :code:`clamp_max` will be reduced to
      :code:`clamp_max` prior to clamping. Defaults to
      :code:`0.95`. Valid values are any float in the range :math:`[0,
      1)`.

   .. py:attribute:: unit_spec

      The :class:`UnitSpec` object containing parameters for the units
      in the layer. Defaults to a :class:`UnitSpec` object with all
      default parameter values.

   .. py:attribute:: log_on_cycle

      An iterable of strings specifying which layer attributes to log
      each cycle. Defaults to an empty iterable, :code:`()`. Valid
      members of the iterable are:

      - :code:`avg_act`, the average layer activation.
      - :code:`avg_net`, the average layer net input.
      - :code:`cos_diff_avg`, the cosine difference between the trial
        plus-phase activation and the minus-phase activation.
      - :code:`fbi`, the layer feedback inhibition.
      - :code:`unit_act`, the activation of each unit.
      - :code:`unit_adapt`, the adaption current of each unit.
      - :code:`unit_gc_i`, the inhibition current in each unit.
      - :code:`unit_i_net`, the net current in each unit.
      - :code:`unit_net_raw`, the unintegrated (raw) net input to each unit.
      - :code:`unit_net`, the integrated net input to each unit.
      - :code:`unit_spike`, the spike status of each unit.
      - :code:`unit_v_m_eq`, the equilibrium membrane potential of
        each unit (like :code:`v_m` but does not reset when a spike
        happens).
      - :code:`unit_v_m`, the membrane potential of each unit.


ProjnSpec
---------

.. py:class:: ProjnSpec()

   Contains parameters for network projections (bundles of unit-to-unit connections.)

   .. py:attribute:: dist

      The probability distrubtion from which connection weights will
      be drawn. Defaults to :class:`lb.Scalar(0.5)`, but can be any
      distribution object listed in :doc:`distributions`.'


   .. py:attribute:: pre_mask

      An iterable of booleans that selects which pre layer units will
      be included in the projection. If the length is less than the
      number of units in the pre layer, it will be tiled. If the
      length is greater, it will be truncated. Defaults to
      :code:`(True, )`, which includes all pre layer units.

   .. py:attribute:: post_mask

      An iterable of booleans that selects which post layer units will
      be included in the projection. If the length is less than the
      number of units in the post layer, it will be tiled. If the
      length is greater, it will be truncated. Defaults to
      :code:`(True, )`, which includes all post layer units.

   .. py:attribute:: sparsity

      Sets the sparsity of the connection. If this is less than
      :code:`1.0`, then :code:`1.0 - sparsity` percent of the
      connections will be randomly disabled. Defaults to
      :code:`1.0`. Valid values are any float in the range :math:`[0,
      1]`.

   .. py:attribute:: projn_type

      Sets the type, or "pattern", of the projection. Defaults to
      :code:`"full"`, which connects every unit in the pre layer to
      every unit in the post layer. Can also be set to
      :code:`"one_to_one"`, which connects every :math:`i_{th}` unit
      in the pre layer to every :math:`i_{th}` unit in the post layer.

   .. py:attribute:: wt_scale_abs

      The absolute net input scaling weight. Simply multiplies net
      input, without taking into account other projections terminating
      in the post layer. Defaults to :code:`1.0`. Valid values are any
      float in the range :math:`[0, \infty)`.

   .. py:attribute:: wt_scale_rel

      The relative net input scaling weight. Multiplies net input, but
      is normalized by the sum of relative net input scaling weights
      across other projections terminating in the same post
      layer. Defaults to :code:`1.0`. Valid values are any float in
      the range :math:`[0, \infty)`.

   .. py:attribute:: lrate

      The learning rate for the projection. Defaults to
      :code:`0.02`. Valid values are any float in the range :math:`[0,
      \infty)`.

   .. py:attribute:: thr_l_mix

      Mixing constant determining the proportion of Hebbian
      learning. A value of :code:`0` denotes no Hebbian learning (so
      the learning will be completely error-driven), and a value of
      :code:`1` denotes only Hebbian learning. Defaults to
      :code:`0.1`. Valid values are any float in the range :math:`[0,
      1]`.

   .. py:attribute:: cos_diff_thr_l_mix

      Boolean flag controlling whether :any:`thr_l_mix` is
      modulated by the post layer's :code:`cos_diff_avg` error
      metric. Defaults to :code:`False`.

   .. py:attribute:: cos_diff_lrate

      Boolean flag controlling whether :any:`lrate` is
      modulated by the post layer's :code:`cos_diff_avg` error
      metric. Defaults to :code:`False`.

   .. py:attribute:: sig_gain

      The gain for the sigmoid function that is used to enhance weight
      contrast before sending net input. Defaults to :code:`6`. Valid
      values are any float in :math:`[0, \infty)`.

   .. py:attribute:: sig_offset

      The offset for the sigmoid function that is used to enhance
      weight contrast before sending net input. Defaults to
      :code:`1`. Valid values are any float.


   .. py:attribute:: log_on_cycle

      An iterable of strings specifying which layer attributes to log
      each cycle. Defaults to an empty iterable, :code:`()`. Valid
      members of the iterable are:

      - :code:`conn_wt`, the sigmoid contrast-enhanced connection weights.
      - :code:`conn_fwt`, the non-contrast-enhanced connection weights.


UnitSpec
---------

.. py:class:: UnitSpec()

   Contains parameters for individual units.

   .. py:attribute:: e_rev_e

      The excitation (net input) reversal potential. Defaults to
      :code:`1`. Valid values are any float.

   .. py:attribute:: e_rev_i

      The inhibitory reversal potential. Defaults to
      :code:`0.25`. Valid values are any float.


   .. py:attribute:: e_rev_l

      The leak reversal potential. Defaults to :code:`0.3`. Valid
      values are any float.

   .. py:attribute:: gc_l

      The leak current, which is always constant. Defaults to
      :code:`0.1`. Valid values are any float.

   .. py:attribute:: spk_thr

      The potential threshold value at which the unit spikes. Defaults
      to :code:`0.5`. Valid values are any float.

   .. py:attribute:: v_m_r

      The potential reset value afte ra spike. Defaults to
      :code:`0.3`. Should be less than :any:`spk_thr`.

   .. py:attribute:: vm_gain

      The adaption current gain from membrane potential. Defaults to
      :code:`0.04`. Valid values are any float in :math:`[0, \infty)`.

   .. py:attribute:: spike_gain

      The adaption current gain from discrete spikes. Defaults to
      :code:`0.00805`. Valid values are any float in :math:`[0, \infty)`.

   .. py:attribute:: net_dt

      The net input integration time constant. Defaults to :code:`1 /
      1.4`. Valid values are any float in :math:`[0, \infty)`.

   .. py:attribute:: vm_dt

      The membrane potential integration time constant. Defaults to
      :code:`1 / 3.3`. Valid values are any float in :math:`[0,
      \infty)`.

   .. py:attribute:: adapt_dt

      The adaption current integration time constant. Defaults to
      :code:`1 / 144`. Valid values are any float in :math:`[0,
      \infty]`.

   .. py:attribute:: syn_tr

      The synaptic transmission efficiency. Defaults to
      :code:`1`. Valid values are any float in :math:`[0, \infty)`.

   .. py:attribute:: act_gain

      The potential gain from clamping (higher values mean a lower
      potential value.) Defaults to :code:`100.0`. Valid values are
      any float in :math:`(0, \infty)`.

   .. py:attribute:: ss_dt

      The supershot learning average integration time
      constant. Defaults to :code:`0.5`. Valid values are any float in
      :math:`[0, \infty)`.

   .. py:attribute:: s_dt

      The short learning average integration time constant. Defaults
      to :code:`0.5`. Valid values are any float in :math:`[0,
      \infty)`.

   .. py:attribute:: m_dt

      The medium learning average integration time constnat. Defaults
      to :code:`0.1`. Valid values are any float in :math:`[0,
      \infty)`.


   .. py:attribute:: l_dn_dt

      The long learning average integration time constant, when it is
      decreasing. Defaults to :code:`2.5`. Valid values are any float
      in :math:`[0, \infty)`.

   .. py:attribute:: l_up_inc

      The increasing long learning average increment
      multiplier. Defaults to :code:`0.2`.  Valid values are any float
      in :math:`[0, \infty)`.


.. py:class:: ValidationError

   Exception raised when a spec contains an invalid parameter value.
