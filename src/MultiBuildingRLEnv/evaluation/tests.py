# A general test suite which
from .pricing_strategy import BasePricingStrategy
import pandas as pd
import numpy as np
from typing import List, Union, Callable
import logging

logger = logging.getLogger(__name__)


class TestPricingStrategy:
    def __init__(
        self,
        consumer_models: list,
        pricing_strategy: BasePricingStrategy,
        test_period: tuple[str, str],
        weather_file: Union[str, List],
        warmup_steps: int = 96 * 5,
        n_epochs: int = 1,
        grid_price_coeff: float = 0.007,
        consumer_prices_min_max_tuple: tuple[float, float] = (0.2, 0.3),
        verbose: bool = True,
    ) -> None:
        self.cms = consumer_models
        self.prix_f = pricing_strategy
        self.start_date, self.end_date = test_period
        self.start_idx, self.stop_idx = (0, 35040)
        self.n_epochs = 1  # number of cycles to repeat the test
        self.n_warmup_steps = warmup_steps
        self.last_powers = [0.0] * len(self.cms)
        self.net_load = []
        self.grid_coeff = grid_price_coeff
        self.verbose = verbose
        self.act_to_price = np.poly1d(
            np.polyfit(
                [-1, 1],
                [consumer_prices_min_max_tuple[0], consumer_prices_min_max_tuple[1]],
                deg=1,
            )
        )
        # Read and set up the weather file
        if isinstance(weather_file, str):
            self._read_weather_file(weather_file)
        else:
            self.index, self.t_amb, self.q_irrad = weather_file
        # Warm up the consumer models
        self._warmup_cons_models()

    def reset(self):
        self.last_powers = [0.0] * len(self.cms)
        self.net_load = []
        for cm in self.cms:
            cm.reset(self.start_idx)

    def _warmup_cons_models(self):
        for t in range(self.n_warmup_steps):
            for ix, cm in enumerate(self.cms):
                self.last_powers[ix] = cm.step(
                    pi=0.0,
                    t_amb=self.t_amb.iloc[t],
                    q_irrad=self.q_irrad.iloc[t],
                    elec_load=0.0,
                )
        for cm in self.cms:
            cm.reset()

    def _read_weather_file(self, weather_file: str):
        self.index = pd.date_range(
            start="01/01/2022", end="01/01/2023", freq="15min", inclusive="left"
        )
        index_h = pd.date_range(
            start="01/01/2022", end="01/01/2023", freq="1h", inclusive="both"
        )
        weather_data = pd.read_csv(
            weather_file
        )  # Contains 1 year of data at hourly resolution
        self.t_amb = (
            pd.Series(
                weather_data["Dry Bulb Temperature {C} "].to_list()
                + [weather_data["Dry Bulb Temperature {C} "].iloc[0]],
                index=index_h,
            )
            .resample("15min", closed="left")
            .mean()
            .interpolate()
        )[
            0:-1
        ]  # degrees C
        self.q_irrad = (
            pd.Series(
                weather_data["Global Horizontal Radiation {Wh/m2} "].to_list()
                + [weather_data["Global Horizontal Radiation {Wh/m2} "].iloc[0]],
                index=index_h,
            )
            .resample("15min")
            .mean()
            .interpolate()
        )[
            0:-1
        ]  # Wh/m2 == W/m2 since hourly data
        start_idx, stop_idx = (
            self.index.get_loc(self.start_date).start,
            self.index.get_loc(self.end_date).stop,
        )
        self.start_idx, self.stop_idx = (start_idx, stop_idx)

    def __call__(
        self,
        keep_in_memory: bool = False,
        callbacks: dict[str, list[tuple[Callable, list]]] = {},
    ) -> dict:
        if self.verbose:
            logger.info(
                f"Testing a cluster of {len(self.cms)} houses from {self.start_date} to {self.end_date}"
            )
        self.reset()
        # Initial callback
        if "on_init" in callbacks:
            for cb in callbacks["on_init"]:
                cb[0](self, *cb[1])
        for epoch in range(self.n_epochs):
            for t in range(self.start_idx, self.stop_idx):
                prix = self.prix_f(
                    t % 96,
                    self.last_powers,
                    self.t_amb.iloc[t],
                    self.q_irrad.iloc[t],
                    info_dict={"tps": self},
                )
                for ix, cm in enumerate(self.cms):
                    self.last_powers[ix] = cm.step(
                        pi=prix[ix],
                        t_amb=self.t_amb.iloc[t],
                        q_irrad=self.q_irrad.iloc[t],
                    )
                self.net_load.append(sum(self.last_powers))
                if "on_step" in callbacks:
                    for cb in callbacks["on_step"]:
                        cb[0](self, *cb[1])
        if self.verbose:
            logger.info("Done! Gathering results.. ")
        results = {}
        # Grid-level metrics
        results["net_load"] = pd.Series(
            self.net_load[0 : (self.stop_idx - self.start_idx)],
            index=self.index[self.start_idx : self.stop_idx],
        )
        # PAR means mean daily par
        nl = results["net_load"]
        metrics = {
            "par": (
                np.array(nl).reshape(-1, 96).max(axis=1)
                / np.array(nl).reshape(-1, 96).mean(axis=1)
            ).mean()
        }
        metrics["agg_expense"] = self.grid_coeff * sum(
            [((p / 1000) ** 2) for p in self.net_load]
        )
        # Consumer-level metrics
        results["cons_bills"] = []
        metrics["agg_income"] = 0.0
        for cm in self.cms:
            cm_load = np.array(cm.net_load) / 4000
            cm_prices = self.act_to_price(np.array(cm.prices))
            cm_bill = cm_load.dot(cm_prices)
            results["cons_bills"].append(cm_bill)
            metrics["agg_income"] += cm_bill
        metrics["mean_cons_bill"] = sum(results["cons_bills"]) / len(
            results["cons_bills"]
        )
        metrics["agg_profit"] = metrics["agg_income"] - metrics["agg_expense"]
        results["metrics"] = metrics
        if "on_end" in callbacks:
            for cb in callbacks["on_end"]:
                cb[0](self, *cb[1])
        # If no need to keep in memory, we can reset the consumer models
        if not keep_in_memory:
            for cm in self.cms:
                cm.reset()
        return results
