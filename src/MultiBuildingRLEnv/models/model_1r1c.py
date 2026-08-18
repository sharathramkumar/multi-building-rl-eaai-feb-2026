# Provide a clean implementation of a 4r2c model
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ModelParams1R1C:
    # Model specific parameters
    c_in: float = 0.0
    r_in: float = 0.0
    # Additional calculation parameters with default values
    f_sol: float = 0.15  # This represents the fraction of phi_sol which reaches the interior of the building
    f_conv: float = 0.09
    f_elec: float = 0.1


@dataclass
class CoolingSystemSpecification:
    # dummy class
    def get_cooling_power(self, t_in: float, t_set: float) -> tuple[float, float]:
        cooling_power = 1000.0
        electric_power = 450.0
        return cooling_power, electric_power


@dataclass
class SimulationDataTracker:
    t_in: list[float] = field(default_factory=list)
    p_ac: list[float] = field(default_factory=list)

    def append(self, _tin, _pac):
        self.t_in.append(_tin)
        self.p_ac.append(_pac)

    def get_pd_series(self, index: pd.Series) -> tuple[pd.Series, pd.Series]:
        return (
            pd.Series(self.t_in, index=index),
            pd.Series(self.p_ac, index=index),
        )


class ThermalModel1R1C:
    def __init__(
        self,
        params: ModelParams1R1C,
        cooling_system: CoolingSystemSpecification,
        init_t_in: float,
        sim_step_size: float = 1.0,  # hours
    ):
        self.params = params
        self.ac = cooling_system
        self.t_in = init_t_in
        self.p_ac = 0.0
        self.ssize = sim_step_size
        # A simple "memory" to track variable evolution over time
        self.history = SimulationDataTracker()
        self.update_history()

    def init_temperatures(self, t_amb: list, q_irrad: list):
        # This initializes the thermal models to settle the temperatures
        for t_a, q_i in zip(t_amb, q_irrad):
            self.step(45.0, t_a, q_i)
        self.reset()

    def update_history(self):
        self.history.append(self.t_in, self.p_ac)

    def reset(self):
        self.history = SimulationDataTracker()
        self.ac.reset()

    def step(
        self, t_set: float, t_amb: float, q_irrad: float, q_elec: float = 0.0
    ) -> float:
        phi_sol = self.params.f_sol * q_irrad
        # Calculate the cooler power and the contributions to different parts
        phi_h, self.p_ac = self.ac.get_cooling_power(self.t_in, t_set)
        phi_h = (phi_h * -1) + (self.params.f_elec * q_elec)  # This is in watts
        # Calculate the temperature evolution
        d_t_in = ((1 / (self.params.r_in * self.params.c_in)) * (t_amb - self.t_in)) + (
            (1 / self.params.c_in) * (phi_h + ((1 - self.params.f_conv) * phi_sol))
        )
        self.t_in = self.t_in + self.ssize * d_t_in
        self.update_history()
        return self.p_ac
