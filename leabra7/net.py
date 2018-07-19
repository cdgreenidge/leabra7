"""A network."""
from typing import Dict
from typing import List
from typing import Sequence

from leabra7 import layer
from leabra7 import log
from leabra7 import events
from leabra7 import projn
from leabra7 import specs


class Net(events.EventListenerMixin):
    """A leabra7 network. This is the main class."""

    def __init__(self) -> None:
        # Each of the following dicts is keyed by the name of the object
        self.objs: Dict[str, events.EventListenerMixin] = {}
        self.layers: Dict[str, layer.Layer] = {}
        self.projns: Dict[str, projn.Projn] = {}
        self.cycle_loggers: List[log.Logger] = []
        self.trial_loggers: List[log.Logger] = []
        self.epoch_loggers: List[log.Logger] = []
        self.batch_loggers: List[log.Logger] = []
        self._cycle_logging = True
        self._trial_logging = True
        self._epoch_logging = True
        self._batch_logging = True

    def _validate_obj_name(self, name: str) -> None:
        """Checks if a name exists within the objects dict.

        Args:
            name: The name to check.

        Raises:
            ValueError: If the name does not exist within the objects dict.
                This is not AssertionError because it is intended to be called
                within user-facing methods.

        """
        if name not in self.objs:
            raise ValueError("No object found with name {0}".format(name))

    def _validate_layer_name(self, name: str) -> None:
        """Checks if a layer name exists.

        Args:
          name: The name of the layer.

        Raises:
          ValueError: If no layer with such a name exists.

        """
        if name not in self.layers:
            raise ValueError(
                "Name {0} does not refer to a layer.".format(name))

    def _get_layer(self, name: str) -> layer.Layer:
        """Gets a layer by name.

        Args:
            name: The name of the layer

        Raises:
            ValueError: If the name does not refer to a layer.
                This is not AssertionError because it is intended to be called
                within user-facing methods.

        """
        self._validate_layer_name(name)
        return self.layers[name]

    def new_layer(self, name: str, size: int,
                  spec: specs.LayerSpec = None) -> None:
        """Adds a new layer to the network.

        Args:
            name: The name of the layer.
            size: How many units the layer should have.
            spec: The layer specification.

        Raises:
            spec.ValidationError: If the spec contains an invalid parameter
                value.

        """
        if spec is not None:
            spec.validate()
        lr = layer.Layer(name, size, spec)
        self.layers[name] = lr
        self.objs[name] = lr

        if lr.spec.log_on_cycle != ():
            logger = log.Logger(lr, lr.spec.log_on_cycle, events.Cycle,
                                events.PauseCycleLog, events.ResumeCycleLog)
            self.cycle_loggers.append(logger)
            self.objs["{0}_cycle_logger".format(name)] = logger
        if lr.spec.log_on_trial != ():
            logger = log.Logger(lr, lr.spec.log_on_trial, events.EndTrial,
                                events.PauseTrialLog, events.ResumeTrialLog)
            self.trial_loggers.append(logger)
            self.objs["{0}_trial_logger".format(name)] = logger
        if lr.spec.log_on_epoch != ():
            logger = log.Logger(lr, lr.spec.log_on_epoch, events.EndEpoch,
                                events.PauseEpochLog, events.ResumeEpochLog)
            self.epoch_loggers.append(logger)
            self.objs["{0}_epoch_logger".format(name)] = logger
        if lr.spec.log_on_batch != ():
            logger = log.Logger(lr, lr.spec.log_on_batch, events.EndBatch,
                                events.PauseBatchLog, events.ResumeBatchLog)
            self.batch_loggers.append(logger)
            self.objs["{0}_batch_logger".format(name)] = logger

    def clamp_layer(self, name: str, acts: Sequence[float]) -> None:
        """Clamps the layer's activations.

        After forcing, the layer's activations will be set to the values
        contained in `acts` and will not change from cycle to cycle.

        Args:
            name: The name of the layer.
            acts: A sequence containing the activations that the layer's
                units will be clamped to. If its length is less than the number
                of units in the layer, it will be tiled. If its length is
                greater, the extra values will be ignored.

        ValueError: If `name` does not match any existing layer name.

        """
        self._validate_layer_name(name)
        self.handle(events.HardClamp(name, acts))

    def unclamp_layer(self, name: str) -> None:
        """Unclamps the layer's activations.

        After unclamping, the layer's activations will be
        updated each cycle.

        """
        self._validate_layer_name(name)
        self.handle(events.Unclamp(name))

    def new_projn(self,
                  name: str,
                  pre: str,
                  post: str,
                  spec: specs.ProjnSpec = None) -> None:
        """Adds a new projection to the network.

        Args:
            name: The name of the projection.
            pre: The name of the sending layer.
            post: The name of the receiving layer.
            spec: The projection specification.

        Raises:
            ValueError: If `pre` or `post` do not match any existing layer
                name.
            spec.ValidationError: If the spec contains an invalid parameter
                value.

        """
        if spec is not None:
            spec.validate()

        pre_lr = self._get_layer(pre)
        post_lr = self._get_layer(post)
        pr = projn.Projn(name, pre_lr, post_lr, spec)
        self.projns[name] = pr
        self.objs[name] = pr

    def _cycle(self) -> None:
        """Cycles the network (triggered by cycle event)."""
        for _, lr in self.layers.items():
            lr.activation_cycle()

        for _, pr in self.projns.items():
            pr.flush()

    def cycle(self) -> None:
        """Cycles the network."""
        self.handle(events.Cycle())

    def minus_phase_cycle(self, num_cycles: int = 50) -> None:
        """Runs a series of cycles for the trial minus phase.

        A minus phase is the trial phase where target values are not clamped
        output layers. Clamping the values on the output layers is the user's
        responsibility.

        Args:
          num_cycles: The number of cycles to run.

        Raises:
          ValueError: If num_cycles is less than 1.

        """
        if num_cycles < 1:
            raise ValueError("Number of cycles must be >= 1.")
        self.handle(events.BeginMinusPhase())
        for _ in range(num_cycles):
            self.handle(events.Cycle())
        self.handle(events.EndMinusPhase())

    def plus_phase_cycle(self, num_cycles: int = 50) -> None:
        """Runs a series of cycles for the trial plus phase.

        A plus phase is the trial phase where target values are clamped on
        output layers. Clamping the values on the output layers is the user's
        responsibility.

        Args:
          num_cycles: The number of cycles to run.

        Raises:
          ValueError: If num_cycles is less than 1.

        """
        if num_cycles < 1:
            raise ValueError("Number of cycles must be >= 1.")
        self.handle(events.BeginPlusPhase())
        for _ in range(num_cycles):
            self.handle(events.Cycle())
        self.handle(events.EndPlusPhase())
        self.handle(events.EndTrial())

    def end_epoch(self) -> None:
        """Signals to the network that an epoch has ended."""
        self.handle(events.EndEpoch())

    def end_batch(self) -> None:
        """Signals to the network that a batch has ended."""
        self.handle(events.EndBatch())

    def learn(self) -> None:
        """Updates projection weights with XCAL learning equation."""
        self.handle(events.Learn())

    @property
    def cycle_logging(self) -> bool:
        return self._cycle_logging

    @property
    def trial_logging(self) -> bool:
        return self._trial_logging

    @property
    def epoch_logging(self) -> bool:
        return self._epoch_logging

    @property
    def batch_logging(self) -> bool:
        return self._batch_logging

    @cycle_logging.setter  #type: ignore
    def cycle_logging(self, run: bool) -> None:
        self._cycle_logging = run
        if self.cycle_logging:
            self.handle(events.ResumeCycleLog())
        else:
            self.handle(events.PauseCycleLog())

    @trial_logging.setter  #type: ignore
    def trial_logging(self, run: bool) -> None:
        self._trial_logging = run
        if self.trial_logging:
            self.handle(events.ResumeTrialLog())
        else:
            self.handle(events.PauseTrialLog())

    @epoch_logging.setter  #type: ignore
    def epoch_logging(self, run: bool) -> None:
        self._epoch_logging = run
        if self.epoch_logging:
            self.handle(events.ResumeEpochLog())
        else:
            self.handle(events.PauseEpochLog())

    @batch_logging.setter  #type: ignore
    def batch_logging(self, run: bool) -> None:
        self._batch_logging = run
        if self.batch_logging:
            self.handle(events.ResumeBatchLog())
        else:
            self.handle(events.PauseBatchLog())

    def logs(self, freq: str, name: str) -> log.Logs:
        """Retrieves logs for an object in the network.

        Args:
            freq: The frequency at which the desired logs were recorded. One
                of `["cycle"]`.
            name: The name of the object for which the logs were recorded.

        Raises:
            ValueError: If the frequency name is invalid, or if no logs were
                recorded for the desired object.

        """
        freq_names = {
            "cycle": self.cycle_loggers,
            "trial": self.trial_loggers,
            "epoch": self.epoch_loggers,
            "batch": self.batch_loggers
        }
        try:
            freq_loggers = freq_names[freq]
        except KeyError:
            raise ValueError("{0} must be one of {1}.".format(
                freq, freq_names.keys()))

        try:
            logger = next(i for i in freq_loggers if i.target_name == name)
        except StopIteration:
            raise ValueError(
                "No logs recorded for object {0}, frequency {1}.".format(
                    name, freq))

        return logger.to_logs()

    def handle(self, event: events.Event) -> None:
        """Overrides events.EventListnerMixin.handle()"""
        if isinstance(event, events.Cycle):
            self._cycle()

        for _, obj in self.objs.items():
            obj.handle(event)
