from dataclasses import dataclass
from random import random


@dataclass
class HysteresisCoolingSystemSpecification:
    # This represents a simple hysteresis controller with the compressor
    # operating at either 50% or 100% capacity
    max_cooling_power: float  # in watts
    cop: float  # watts/watts
    deadband: float = 0.5  # deg C
    # The following is a dynamic parameter which tracks if the cooler is on or off
    status: bool = False

    def get_cooling_power(
        self, t_in: float, t_set: float, t_amb: float = None
    ) -> tuple[float, float]:
        # Hysteresis controller
        # See if the cooler status needs to change
        if self.status and (t_in <= (t_set - self.deadband)):
            self.status = False
        if not self.status and (t_in >= (t_set + self.deadband)):
            self.status = True
        # The controller power should depend on the difference between t_in and t_set
        cooling_power = 0.0
        if t_in - t_set > 0.5 * self.deadband:
            cooling_power = self.max_cooling_power
        elif t_in - t_set >= 0.25 * self.deadband:
            cooling_power = 0.5 * self.max_cooling_power
        cooling_power = (
            cooling_power if self.status else 0.0
        )  # Higher positive -> more cooling
        electric_power = abs(cooling_power / self.cop)
        return cooling_power, electric_power


@dataclass
class KMovingAverageBoundController:
    # This controller goes to the default bound if a condition based on a k-step moving average
    # is violated
    # AC specification
    max_cooling_power: float  # in watts
    cop: float  # watts/watts
    deadband: float = 0.2  # deg C
    # The following is a dynamic parameter which tracks if the cooler is on or off
    status: bool = False
    # The following are the specifications for the moving average calculation
    default_setpoint: float = 25.0
    k: int = 8
    threshold: float = (
        0.35  # k-MA of t_in, The higher this parameter, the more flexible the user
    )
    # The following are internal attributes
    _temp_hist = [0.0] * k
    _n_overrides = 0
    override_history = []

    def get_cooling_power(
        self, t_in: float, t_set: float, t_amb: float = None
    ) -> tuple[float, float]:
        # Check if the t_set needs to be overridden
        self._temp_hist = self._temp_hist[1:] + [t_in - self.default_setpoint]
        if (sum(self._temp_hist) / self.k) >= self.threshold:
            if (t_set - self.default_setpoint) > 0.1:
                self._n_overrides += 1
                self.override_history.append(1)
            else:
                self.override_history.append(0)
            t_set = self.default_setpoint
        else:
            self.override_history.append(0)
        # Hysteresis controller
        # See if the cooler status needs to change
        if self.status and (t_in <= (t_set - self.deadband)):
            self.status = False
        if not self.status and (t_in >= (t_set + self.deadband)):
            self.status = True
        # The controller power should depend on the difference between t_in and t_set
        # cooling_power = 0.0
        # if t_in - t_set > self.deadband:
        #     cooling_power = self.max_cooling_power
        # elif t_in - t_set > -self.deadband:
        #     cooling_power = 0.5 * self.max_cooling_power
        # cooling_power = (
        #     cooling_power if self.status else 0.0
        # )  # Higher positive -> more cooling
        cooling_power = self.max_cooling_power if self.status else 0.0
        electric_power = abs(cooling_power / self.cop)
        return cooling_power, electric_power

    def reset(self):
        self._temp_hist = [0.0] * self.k
        self._n_overrides = 0
        self.override_history = []


@dataclass
class KMovingAverageBoundControllerFixed:
    # This controller goes to the default bound if a condition based on a k-step moving average
    # is violated
    # AC specification
    max_cooling_power: float  # in watts
    cop: float  # watts/watts
    deadband: float = 0.2  # deg C
    # The following is a dynamic parameter which tracks if the cooler is on or off
    status: bool = False
    # The following are the specifications for the moving average calculation
    default_setpoint: float = 25.0
    k: int = 8
    threshold: float = (
        0.35  # k-MA of t_in, The higher this parameter, the more flexible the user
    )
    # The following are internal attributes
    _temp_hist = [0.0] * k
    _n_overrides = 0
    override_history = []

    def copy_from(self, other: KMovingAverageBoundController):
        self.max_cooling_power = other.max_cooling_power
        self.cop = other.cop
        self.deadband = other.deadband
        self.default_setpoint = other.default_setpoint
        self.k = other.k
        self.threshold = other.threshold
        return self

    def get_cooling_power(
        self, t_in: float, t_set: float, t_amb: float = None
    ) -> tuple[float, float]:
        # Check if the t_set needs to be overridden
        self._temp_hist = self._temp_hist[1:] + [t_in - self.default_setpoint]
        if (sum(self._temp_hist) / self.k) >= self.threshold:
            if (t_set - self.default_setpoint) > 0.1:
                self._n_overrides += 1
                self.override_history.append(1)
            else:
                self.override_history.append(0)
            t_set = self.default_setpoint
        else:
            self.override_history.append(0)
        # Hysteresis controller
        # See if the cooler status needs to change
        if self.status and (t_in <= (t_set - self.deadband)):
            self.status = False
        if not self.status and (t_in >= (t_set + self.deadband)):
            self.status = True
        # The controller power should depend on the difference between t_in and t_set
        cooling_power = (
            self.max_cooling_power if self.status else 0.0
        )  # Higher positive -> more cooling
        electric_power = abs(cooling_power / self.cop)
        return cooling_power, electric_power

    def reset(self):
        self._temp_hist = [0.0] * self.k
        self._n_overrides = 0
        self.override_history = []


@dataclass
class StochasticKMovingAverageBoundControllerFixed:
    # This controller goes to the default bound if a condition based on a k-step moving average
    # is violated, but in a stochastic manner
    # AC specification
    max_cooling_power: float  # in watts
    cop: float  # watts/watts
    deadband: float = 0.2  # deg C
    # The following is a dynamic parameter which tracks if the cooler is on or off
    status: bool = False
    # The following are the specifications for the moving average calculation
    default_setpoint: float = 25.0
    k: int = 8
    threshold: float = (
        0.35  # k-MA of t_in, The higher this parameter, the more flexible the user
    )
    # The following are internal attributes
    _temp_hist = [0.0] * k
    _n_overrides = 0
    override_history = []

    @staticmethod
    def copy_from(other: KMovingAverageBoundController):
        return StochasticKMovingAverageBoundControllerFixed(
            other.max_cooling_power,
            other.cop,
            other.deadband,
            default_setpoint=other.default_setpoint,
            k=other.k,
            threshold=other.threshold,
        )

    def get_cooling_power(
        self, t_in: float, t_set: float, t_amb: float = None
    ) -> tuple[float, float]:
        # Check if the t_set needs to be overridden
        self._temp_hist = self._temp_hist[1:] + [t_in - self.default_setpoint]
        discomfort_metric = sum(self._temp_hist) / self.k
        will_override = random() < (discomfort_metric / self.threshold)
        if will_override:
            self._n_overrides += 1
            self.override_history.append(1)
            t_set = self.default_setpoint
        else:
            self.override_history.append(0)
        # Hysteresis controller
        # See if the cooler status needs to change
        if self.status and (t_in <= (t_set - self.deadband)):
            self.status = False
        if not self.status and (t_in >= (t_set + self.deadband)):
            self.status = True
        # The controller power should depend on the difference between t_in and t_set
        cooling_power = (
            self.max_cooling_power if self.status else 0.0
        )  # Higher positive -> more cooling
        electric_power = abs(cooling_power / self.cop)
        return cooling_power, electric_power

    def reset(self):
        self._temp_hist = [0.0] * self.k
        self._n_overrides = 0
        self.override_history = []


@dataclass
class StochasticMultipleFactorOverrideModelSarran:
    # This controller goes to the default bound if a condition based on a k-step moving average
    # is violated, but in a stochastic manner
    # AC specification
    max_cooling_power: float  # in watts
    cop: float  # watts/watts
    deadband: float = 0.2  # deg C
    # The following is a dynamic parameter which tracks if the cooler is on or off
    status: bool = False
    # The following are the specifications for the moving average calculation
    default_setpoint: float = 25.0
    k: int = 8
    interaction_level: float = (
        0.0  # This value, from 0 to 1, quantifies how frequently the user interacts with their thermostat
    )
    # The following are internal attributes
    _temp_hist = [0.0] * k
    _n_overrides = 0
    override_history = []
    leaf_choices = []

    @staticmethod
    def copy_from(other: KMovingAverageBoundController):
        return StochasticMultipleFactorOverrideModelSarran(
            other.max_cooling_power,
            other.cop,
            other.deadband,
            default_setpoint=other.default_setpoint,
            k=other.k,
            interaction_level=random(),
        )

    def get_cooling_power(
        self, t_in: float, t_set: float, t_amb: float = None
    ) -> tuple[float, float]:
        # Check if the t_set needs to be overridden
        # Decision tree based on Sarran et al 2021
        override_probability = 0.0
        if self.interaction_level <= 0.33:
            # less interaction branch
            if t_amb < 27.0:
                if (abs(t_in - self.default_setpoint)) < 1.0:
                    override_probability = 0.15
                    self.leaf_choices.append(0)
                else:
                    override_probability = 0.2
                    self.leaf_choices.append(1)
            else:
                if (abs(t_in - t_set)) > 1.0:
                    override_probability = 0.40
                    self.leaf_choices.append(2)
                else:
                    override_probability = 0.15
                    self.leaf_choices.append(3)
        elif self.interaction_level <= 0.66:
            # med interaction
            if t_amb < 26.5:
                override_probability = 0.25
                self.leaf_choices.append(4)
            else:
                override_probability = 0.35
                self.leaf_choices.append(5)
        else:
            # high interaction
            if sum(self.override_history[-self.k :]) / self.k > 0.6:
                override_probability = 0.85
                self.leaf_choices.append(6)
            else:
                override_probability = 0.35
                self.leaf_choices.append(7)
        will_override = random() < override_probability
        if will_override:
            self._n_overrides += 1
            self.override_history.append(1)
            t_set = self.default_setpoint
        else:
            self.override_history.append(0)
        # Hysteresis controller
        # See if the cooler status needs to change
        if self.status and (t_in <= (t_set - self.deadband)):
            self.status = False
        if not self.status and (t_in >= (t_set + self.deadband)):
            self.status = True
        # The controller power should depend on the difference between t_in and t_set
        cooling_power = (
            self.max_cooling_power if self.status else 0.0
        )  # Higher positive -> more cooling
        electric_power = abs(cooling_power / self.cop)
        return cooling_power, electric_power

    def reset(self):
        self._temp_hist = [0.0] * self.k
        self._n_overrides = 0
        self.override_history = []


@dataclass
class VariableSpeedKMovingAverageBoundController:
    # This controller goes to the default bound if a condition based on a k-step moving average
    # is violated
    # AC specification
    max_cooling_power: float  # in watts
    cop: float  # watts/watts
    deadband: float = 0.5  # deg C
    # The following is a dynamic parameter which tracks if the cooler is on or off
    status: bool = False
    # The following are the specifications for the moving average calculation
    default_setpoint: float = 25.0
    k: int = 8
    threshold: float = (
        0.35  # k-MA of t_in, The higher this parameter, the more flexible the user
    )
    # The following are internal attributes
    _temp_hist = [0.0] * k

    def get_cooling_power(
        self, t_in: float, t_set: float, t_amb: float = None
    ) -> tuple[float, float]:
        # Check if the t_set needs to be overridden
        self._temp_hist = self._temp_hist[1:] + [t_in - self.default_setpoint]
        if (sum(self._temp_hist) / self.k) >= self.threshold:
            t_set = self.default_setpoint
        # Hysteresis controller
        # See if the cooler status needs to change
        if self.status and (t_in <= (t_set - self.deadband)):
            self.status = False
        if not self.status and (t_in >= (t_set + self.deadband)):
            self.status = True
        # The controller power should depend on the difference between t_in and t_set
        deltaT = t_in - t_set
        cooling_power = (2 * self.deadband / self.max_cooling_power) * (
            deltaT + self.deadband
        )
        cooling_power = (
            cooling_power if self.status else 0.0
        )  # Higher positive -> more cooling
        electric_power = abs(cooling_power / self.cop)
        return cooling_power, electric_power
