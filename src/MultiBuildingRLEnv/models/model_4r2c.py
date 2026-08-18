# Provide a clean implementation of a 4r2c model
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ModelParams4R2C:
    # Model specific parameters
    c_in: float = 0.0
    c_e: float = 0.0
    r_in_ia: float = 0.0
    r_ia_a: float = 0.0
    r_ia_e: float = 0.0
    r_e_a: float = 0.0
    # Additional calculation parameters with default values
    f_sol: float = (
        0.15  # This represents the fraction of phi_sol which reaches the interior of the building
    )
    f_heat_rad: float = 0.2
    f_heat_rad_ext: float = 1.5 / 2.5
    f_conv: float = 0.09
    # This parameters affects how t_amb_eq is calculated
    f_shading: float = 0.5
    # The heat gain offered by appliances in the building
    f_elec: float = 0.1
    # A correction term to account for inaccuracies
    q_correction: float = 0.0


@dataclass
class CoolingSystemSpecification:
    # dummy class
    def get_cooling_power(
        self, t_in: float, t_set: float, t_amb: float
    ) -> tuple[float, float]:
        cooling_power = 1000.0
        electric_power = 450.0
        return cooling_power, electric_power


# Some pre-calculated defaults
ModelParams_3RoomApartment_ConcreteWall_70sqm = ModelParams4R2C(
    c_e=3700, c_in=3000, r_in_ia=0.007, r_ia_a=0.02, r_ia_e=0.001, r_e_a=0.001
)

ModelParams_4RoomApartment_ConcreteWall_100sqm = ModelParams4R2C(
    c_e=4300, c_in=3500, r_in_ia=0.005, r_ia_a=0.015, r_ia_e=0.0007, r_e_a=0.001
)


@dataclass
class SimulationDataTracker:
    t_in: list[float] = field(default_factory=list)
    t_e: list[float] = field(default_factory=list)
    t_ia: list[float] = field(default_factory=list)
    p_ac: list[float] = field(default_factory=list)

    def append(self, _tin, _te, _tia, _pac):
        self.t_in.append(_tin)
        self.t_e.append(_te)
        self.t_ia.append(_tia)
        self.p_ac.append(_pac)

    def get_pd_series(
        self, index: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        n_to_skip = max(len(self.t_in) - len(index), 0)
        return (
            pd.Series(self.t_in[n_to_skip:], index=index),
            pd.Series(self.t_e[n_to_skip:], index=index),
            pd.Series(self.t_ia[n_to_skip:], index=index),
            pd.Series(self.p_ac[n_to_skip:], index=index),
        )


class ThermalModel4R2C:
    def __init__(
        self,
        params: ModelParams4R2C,
        cooling_system: CoolingSystemSpecification,
        init_t_in: float,
        init_t_e: float,
        init_t_ia: float,
        sim_step_size: float = 1.0,  # hours
    ):
        self.params = params
        self.ac = cooling_system
        self.t_in = init_t_in
        self.t_e = init_t_e
        self.t_ia = init_t_ia
        self.p_ac = 0.0
        self.ssize = sim_step_size
        # A simple "memory" to track variable evolution over time
        self.history = SimulationDataTracker()
        self.update_history()

    def update_history(self):
        self.history.append(self.t_in, self.t_e, self.t_ia, self.p_ac)

    def init_temperatures(self, t_amb: list, q_irrad: list):
        # This initializes the thermal models to settle the temperatures
        for t_a, q_i in zip(t_amb, q_irrad):
            self.step(45.0, t_a, q_i)
        self.reset()

    def reset(self):
        self.history = SimulationDataTracker()
        self.ac.reset()

    def step(
        self,
        t_set: float,
        t_amb: float,
        q_irrad: float,
        q_elec: float = 0.0,
        phi_h: float = None,
    ) -> float:
        phi_sol = self.params.f_sol * q_irrad
        t_eq_amb = t_amb + (q_irrad * (0.5 / 25)) * self.params.f_shading
        # Calculate the cooler power and the contributions to different parts
        if phi_h is not None:
            self.p_ac = abs(phi_h)
        else:
            phi_h, self.p_ac = self.ac.get_cooling_power(self.t_in, t_set, t_amb)
        phi_h = (phi_h * -1) + (self.params.f_elec * q_elec)  # This is in watts
        phi_h_ia = phi_h * (1 - self.params.f_heat_rad)
        phi_h_in = (phi_h - phi_h_ia) * (
            1 - self.params.f_heat_rad_ext
        ) + self.params.q_correction
        phi_h_e = (phi_h - phi_h_ia) * self.params.f_heat_rad_ext
        # Calculate the temperature evolution
        d_t_in = (
            (1 / (self.params.r_in_ia * self.params.c_in)) * (self.t_ia - self.t_in)
        ) + ((1 / self.params.c_in) * (phi_h_in + ((1 - self.params.f_conv) * phi_sol)))
        d_t_e = (
            ((1 / (self.params.r_ia_e * self.params.c_e)) * (self.t_ia - self.t_e))
            + ((1 / (self.params.r_e_a * self.params.c_e)) * (t_eq_amb - self.t_e))
            + ((1 / self.params.c_e) * (phi_h_e))
        )
        self.t_in = self.t_in + self.ssize * d_t_in
        self.t_e = self.t_e + self.ssize * d_t_e
        t_ia_dnm = (
            (1 / self.params.r_in_ia)
            + (1 / self.params.r_ia_e)
            + (1 / self.params.r_ia_a)
        )
        t_ia_num = (
            (self.t_in / self.params.r_in_ia)
            + (self.t_e / self.params.r_ia_e)
            + (t_amb / self.params.r_ia_a)
            + (self.params.f_conv * phi_sol)
            + phi_h_ia
        )
        self.t_ia = t_ia_num / t_ia_dnm
        self.update_history()
        return self.p_ac
