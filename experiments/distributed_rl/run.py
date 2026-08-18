from MultiBuildingRLEnv.envs.rl_envs_dev import (
    SingleConsTrainingEnvWithGlobalFullState,
    ClusterTrainingEnvWithGlobalFullState,
)
from MultiBuildingRLEnv.envs.rl_envs_dev.ClusterTrainingEnvWithGlobalFullState import (
    ClusterTrainingEnvWithGlobalFullStateCallback,
)
from MultiBuildingRLEnv.evaluation.pricing_strategy import (
    StaticPricingStrategy,
    IndividualRLPricingStrategyWithGlobalFullState,
)
from MultiBuildingRLEnv.evaluation.tests import TestPricingStrategy
from MultiBuildingRLEnv.models.gen_models import (
    RepeatingGenerator,
    ClusterRepeatingGenerator,
)
from MultiBuildingRLEnv.policies.SplitCriticPolicy import SplitInputActorCriticPolicy
from stable_baselines3.ppo import PPO
import dill
import numpy as np
import pandas as pd
from copy import deepcopy
import os, logging, sys
from itertools import compress

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
train_cons_models_file = find_cons_models_path(
    data_root_path, data_cfg, cfg, is_train=True
)
weather_file = os.path.join(data_root_path, data_cfg["weather_data"])

logger.info(f"Using train cons model file : {train_cons_models_file}")

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

with open(train_cons_models_file, "rb") as ff:
    train_cons_models = dill.load(ff)

train_params = cfg["rl_training_params"]
actor_network_sizes = train_params["actor_network_sizes"]
seeds = train_params["seeds"]
n_train_steps = train_params["n_train_steps"]

# Note the cluster indices
cluster_indices = np.arange(len(cons_models))
uniq_clusters = set(cluster_indices)

# Collect test data from the system
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
non_hv_loads_matrix = np.vstack(
    [non_hv_load(cm, index) for cm in fix_tps.cms]
)  # A matrix of shape (N_buildings, N_training_points)
logging.debug(f"Got a load generator with matrix of shape {non_hv_loads_matrix.shape}")


# Prepare the environments which will be used for normalization in testing
def get_norm_env(cm):
    return SingleConsTrainingEnvWithGlobalFullState(
        fitted_cons_model=deepcopy(cm),
        load_generator=RepeatingGenerator(cm.net_load),
        weather_generator=(
            RepeatingGenerator(
                fix_tps.t_amb[fix_tps.start_idx : fix_tps.stop_idx].to_list()
            ),
            RepeatingGenerator(
                fix_tps.q_irrad[fix_tps.start_idx : fix_tps.stop_idx].to_list()
            ),
        ),
        global_load_generator=RepeatingGenerator(res_fix["net_load"].to_numpy()),
        n_prev_load=train_params["n_prev_load"],
        n_prev_weather=train_params["n_prev_weather"],
    )


rl_norm_envs = [get_norm_env(cm) for cm in fix_tps.cms]

cm_tag = f"cons_{len(cons_models)}"
os.makedirs(cm_tag, exist_ok=True)


def get_env(cluster_idx):
    load_gen = ClusterRepeatingGenerator(
        non_hv_loads_matrix[np.where(cluster_indices == cluster_idx), :][0]
    )
    cons_models = [
        item
        for item in compress(
            list(train_cons_models.values()), cluster_indices == cluster_idx
        )
    ]
    logging.debug(
        f"For cluster {cluster_idx}, load gen has shape {load_gen.seqs.shape} and there are {len(cons_models)} models."
    )
    return ClusterTrainingEnvWithGlobalFullState(
        cons_models=cons_models,
        load_generators=load_gen,
        global_load_generator=RepeatingGenerator(fix_tps.net_load),
        weather_generator=(
            RepeatingGenerator(
                fix_tps.t_amb[fix_tps.start_idx : fix_tps.stop_idx].to_list()
            ),
            RepeatingGenerator(
                fix_tps.q_irrad[fix_tps.start_idx : fix_tps.stop_idx].to_list()
            ),
        ),
        n_prev_load=train_params["n_prev_load"],
        n_prev_weather=train_params["n_prev_weather"],
        cons_factor_offset=train_params["cons_factor_offset"],
    )


for net_size in actor_network_sizes:
    for seed in seeds:
        logger.info(
            f"Starting training for agent with actor network size {net_size}, seed {seed}"
        )
        rl_models_cluster = {}
        for cluster in uniq_clusters:
            # Create the RL environment
            this_expt_tag = (
                f"{expt_tag}_{cm_tag}_net_{net_size[0]}_{net_size[1]}_seed_{seed}"
            )
            logger.info(
                f"Run 'tensorboard --logdir {os.path.join(os.getcwd(), 'tb_logs')}' to monitor training, "
                f"look for the run tagged '{this_expt_tag}'"
            )
            env = get_env(cluster)
            policy_kwargs = dict(
                net_arch=dict(pi=net_size, vf=train_params["critic_network_size"]),
                n_features_to_strip=2 * env.n_prev_tamb,
            )  # This will strip the indoor temperature and the override status from the policy network
            model = PPO(
                SplitInputActorCriticPolicy,
                env,
                gamma=train_params["gamma"],
                learning_rate=train_params["learning_rate"],
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(os.getcwd(), "tb_logs"),
                seed=seed,
                device="cpu",
            )
            model.learn(
                int(n_train_steps),
                callback=ClusterTrainingEnvWithGlobalFullStateCallback(),
            )
            model_save_path = os.path.join(
                os.getcwd(), "models", expt_tag + f"_{seed}_cons_{cluster}.mdl"
            )
            model.save(model_save_path)
            logging.info(f"Done! Saved model to {model_save_path}")
            rl_models_cluster[cluster] = model_save_path

        # At this stage, all the models for this seed are ready
        logging.info(f"Testing the agent in the real system..")
        rl_models = [PPO.load(rl_models_cluster[c]) for c in cluster_indices]
        t_rl_ps = IndividualRLPricingStrategyWithGlobalFullState(
            rl_models, rl_norm_envs
        )
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
                    "rl_envs": rl_norm_envs,
                },
                ff,
            )

logging.info(f"Succesfully completed!")
