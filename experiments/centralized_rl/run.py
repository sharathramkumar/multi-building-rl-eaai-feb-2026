from MultiBuildingRLEnv.envs.rl_envs_dev.MultiConsSingleRLTrainingEnv import (
    MultiConsSingleRLTrainingEnv,
)
import pandas as pd
import dill
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from MultiBuildingRLEnv.evaluation.pricing_strategy import (
    StaticPricingStrategy,
    MultiRLPricingStrategy,
)
from MultiBuildingRLEnv.evaluation.tests import TestPricingStrategy
from MultiBuildingRLEnv.models.gen_models import RepeatingGenerator
import logging, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expt_utils import *

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s - %(asctime)s :  %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Load config
DATA_CONFIG_PATH = "../../configs/data_paths.yaml"
data_cfg, data_root_path, cfg = load_configs(DATA_CONFIG_PATH)
cons_models_file = find_cons_models_path(data_root_path, data_cfg, cfg)
weather_file = os.path.join(data_root_path, data_cfg["weather_data"])

expt_tag = cfg["expt_tag"]
start_date = cfg["train_period"]["start"]
end_date = cfg["train_period"]["end"]
t_start_date = cfg["test_period"]["start"]
t_end_date = cfg["test_period"]["end"]

logger.info(f"Starting run with tag : {expt_tag}!")

end_date_p1 = next_day(end_date)
index = pd.date_range(start=start_date, end=end_date_p1, freq="15min", inclusive="left")
with open(cons_models_file, "rb") as ff:
    cons_models = dill.load(ff)

train_params = cfg["rl_training_params"]
actor_network_sizes = train_params["actor_network_sizes"]
seeds = train_params["seeds"]
n_train_steps = train_params["n_train_steps"]

fix_ps = StaticPricingStrategy(0.0)
fix_tps = TestPricingStrategy(
    deepcopy(cons_models),
    fix_ps,
    (start_date, end_date),
    weather_file,
)
res_fix = fix_tps(True)

non_hv_load = lambda cm, ind: (
    cm.get_pd_series(ind) - cm.th_mod.history.get_pd_series(ind)[-1]
).to_numpy()
load_gens = [RepeatingGenerator(non_hv_load(cm, index)) for cm in fix_tps.cms]

env = MultiConsSingleRLTrainingEnv(
    fitted_cons_models=cons_models,
    load_generators=load_gens,
    weather_generators=(
        RepeatingGenerator(
            fix_tps.t_amb[fix_tps.start_idx : fix_tps.stop_idx].to_list()
        ),
        RepeatingGenerator(
            fix_tps.q_irrad[fix_tps.start_idx : fix_tps.stop_idx].to_list()
        ),
    ),
    n_prev_load=train_params["n_prev_load"],
    n_prev_weather=train_params["n_prev_weather"],
)

cm_tag = f"cons_{len(cons_models)}"
os.makedirs(cm_tag, exist_ok=True)

for net_size in actor_network_sizes:
    for seed in seeds:
        logger.info(
            f"Starting training for agent with actor network size {net_size}, seed {seed}"
        )
        this_expt_tag = (
            f"{expt_tag}_{cm_tag}_net_{net_size[0]}_{net_size[1]}_seed_{seed}"
        )
        logger.info(
            f"Run 'tensorboard --logdir {os.path.join(os.getcwd(), 'tb_logs')}' to monitor training, "
            f"look for the run tagged '{this_expt_tag}'"
        )
        policy_kwargs = dict(
            net_arch=dict(pi=net_size, vf=train_params["critic_network_size"]),
        )
        model = PPO(
            "MlpPolicy",
            env,
            gamma=train_params["gamma"],
            learning_rate=train_params["learning_rate"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(os.getcwd(), "tb_logs"),
            seed=seed,
            device="cpu",
        )
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=int(n_train_steps / 15),
            save_path="./models/checkpoints/",
            name_prefix=this_expt_tag,
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        model.learn(int(n_train_steps), callback=checkpoint_callback)
        model_save_path = os.path.join(os.getcwd(), "models", expt_tag + f"_{seed}.mdl")
        model.save(model_save_path)
        logging.info(f"Done! Saved model to {model_save_path}")

        # At this stage, all the models for this seed are ready
        logging.info(f"Testing the agent in the real system..")
        t_rl_ps = MultiRLPricingStrategy(model, env)
        t_rl_tps = TestPricingStrategy(
            cons_models,
            t_rl_ps,
            (t_start_date, t_end_date),
            weather_file,
        )
        t_res_rl = t_rl_tps(True)
        logging.info(f"Saving results..")
        with open(f"{cm_tag}/res_rl_{expt_tag}_seed_{seed}.dat", "wb+") as ff:
            dill.dump(
                {
                    "cons_models": cons_models,
                    "res": t_res_rl,
                },
                ff,
            )

logging.info(f"Succesfully completed!")
