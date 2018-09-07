The Network Object
==================

.. toctree::
   :maxdepth: 2

.. module:: leabra7


.. py:class:: Net()

   The :class:`Net` object is the primary point of interaction for
   scripts that use **leabra7**. It provides methods to construct the
   network, advance the network in time, and collect output data.

   .. py:method:: load(filename: str) -> None:

      Loads the network from a pickle file, overwriting the current
      network configuration. This function is insecure and could
      potentially execute arbitrary code; be sure not to load malicious
      or untrusted files.

      :param filename: The file from which to load the network.

   .. py:method:: save(filename: str) -> None:

      Saves the network as a pickle file.

      :param filename: Where to save the network.

   .. py:method:: new_layer(name: str, size: int, spec: lb.LayerSpec=None) -> None:

      Adds a new layer to the network.

      :param name: The name of the layer.
      :param size: How many units the layer should have.
      :param spec: The layer spec. If :code:`None`, the default layer
		   spec will be used.
      :raises ValidationError: If the spec contains an invalid
				  parameter value.

   .. py:method:: new_projn(name: str, pre: str, post: str, size: int, spec: lb.ProjnSpec=None) -> None:

      Adds a new projection to the network.

      :param name: The name of the projection.
      :param pre: The name of the sending layer.
      :param post: The name of the receiving layer.
      :param spec: The projection spec. If :code:`None`, the default spec will be
		   used.
      :raises ValueError: If :code:`pre` or :code:`post` do not match
			  any existing layer names.
      :raises ValidationError: If the spec contains an invalid parameter value.


   .. py:method:: clamp_layer(name: str, acts: Sequence[float]) -> None:

      Clamps layer's activations to the specified values, so that they do
      not change from cycle to cycle.

      :param name: The name of the layer to clamp.
      :param acts: A sequence containing the activations to which the
		   layer's units will be clamped. If its length is less
		   than the number of units in the layer, it will be
		   tiled. If its length is greater, the extra values will be ignored.
      :raises ValueError: If :code:`name` does not match any existing layer name.

   .. py:method:: unclamp_layer(name: str) -> None:

      Unclamps a previously-clamped layer. If the layer is not clamped,
      then nothing happens.

      :param name: The name of the layer to unclamp.


   .. py:method:: cycle() -> None:

      Cycles the network.

   .. py:method:: minus_phase_cycle(num_cycles: int = 50) -> None:

      Runs a series of cycles for the trial minus phase, signaling the
      network to compute the appropriate metrics at the beginning and end
      of the phase.

      A minus phase is the trial phase where input patterns are clamped
      to the input layers, but output patterns are not clamped to the
      output layers. This clamping is the user responsibility.

      :param num_cycles: The number of cycles in the minus phase.
      :raises ValueError: If :code:`num_cycles` is less than 1.

   .. py:method:: plus_phase_cycle(num_cycles: int = 25) -> None:

      Runs a series of cycles for the trial plus phase, which is like the
      minus phase except that target values are clamped on the output
      layers.

      :param num_cycles: The number of cycles in the plus phase.
      :raises ValueError: If :code:`num_cycles` is less than 1.

   .. py:method:: learn() -> None:

      Updates the projection weights with the XCAL learning equation.

   .. py:method:: end_epoch() -> None:

      Signals the network that an epoch (one pass through the training
      data) has ended. This must be called by the user at the end of
      every epoch.

   .. py:method:: end_batch() -> None:

      Signals the network that a batch has ended (typically, a batch
      is a series of epochs). This must be called by the user at the end of
      every batch.

   .. py:method:: observe(name: str, attr: str) -> pd.DataFrame:

      This is like logging, but it only returns the current object state.
      There is no history. If you do not require historical observations,
      use this to avoid the performance penalty of logging.

      :param name: The name of the object to observe.
      :param attr: The name of the attribute to observe. This can be
                   any loggable attribute. Check the object's
                   associated spec object for a list of valid loggable
                   attributes.
      :raises ValueError: If the object does not exist, does not
                          support observations, or if the attribute is
                          not a valid loggable attribute.
      :returns: A Pandas dataframe containing the observation result.

   .. py:method:: logs(freq: str, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

      Retrieves logs (observations recorded over time) for an object
      in the network. Which logs should be recorded is specified in
      :doc:`Spec<specs>` objects at network creation, using the
      parameters :code:`"log_on_cycle"`, :code:`"log_on_trial"`, etc.

      :param freq: The frequency at which the desired logs were
                   recorded. One of :code:`["cycle", "trial", "epoch",
                   "batch"]`.
      :param name: The name of the object for which the logs were recorded.
      :raises ValueError: If the frequency name is invalid, or if no
                          logs were recorded for the desired object.
      :returns: A tuple of Pandas dataframes. The first element of the
                tuple contains the logs for "whole" attributes, which
                are attributes of the object itself, like a layer's
                :code:`"avg_act"`. The second element of the tuple
                contains the logs for "parts" attributes, which are
                attributes of the object's constitutents, like a
                layer's :code:`"unit_act"` attribute. For layers,
                parts attributes pertain to the units, and for
                projections, parts attributes pertain to the
                connections.

   .. py:method:: pause_logging(freq: str=None) -> None:

      Pauses logging in the network, if any logging is enabled. This is
      typically done for performance reasons; logging is quite slow.

      :param freq: The frequency for which to pause the logging, one of
		   :code:`"cycle"`, :code:`"trial"`,, :code:`"epoch"`,
		   or :code:`"batch"`. If :code:`None`, pauses for all
		   frequencies.
      :raises ValueError: If no frequency with name :code:`freq` exists.

   .. py:method:: resume_logging(freq: str=None) -> None:

      Resumes logging in the network.

      :param freq: The frequency for which to resume the logging, one of
		   :code:`"cycle"`, :code:`"trial"`,, :code:`"epoch"`,
		   or :code:`"batch"`. If :code:`None`, pauses for all
		   frequencies.
      :raises ValueError: If no frequency with name :code:`freq` exists.
