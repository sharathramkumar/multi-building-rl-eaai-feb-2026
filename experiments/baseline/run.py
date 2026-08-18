from MultiBuildingRLEnv.evaluation.pricing_strategy import StaticPricingStrategy
from MultiBuildingRLEnv.evaluation.tests import TestPricingStrategy
import dill
from copy import deepcopy
import os, logging, sys

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
t_start_date = cfg["test_period"]["start"]
t_end_date = cfg["test_period"]["end"]

logger.info(f"Starting run with tag : {expt_tag}!")

logger.info("Loading consumer models..")
with open(cons_models_file, "rb") as ff:
    cons_models = dill.load(ff)
logger.info("Done!")

cm_tag = f"cons_{len(cons_models)}"
os.makedirs(cm_tag, exist_ok=True)

# Collect test data from the system
logger.info("Running baseline experiment.. ")
fix_ps = StaticPricingStrategy(0.0)
fix_tps = TestPricingStrategy(
    deepcopy(cons_models),
    fix_ps,
    (t_start_date, t_end_date),
    weather_file,
)

res_fix = fix_tps(True)
logger.info("Done!")
logger.info(f"Saving results..")
with open(f"{cm_tag}/res_{expt_tag}.dat", "wb+") as ff:
    dill.dump(
        {
            "cons_models": fix_tps.cms,
            "res": res_fix,
        },
        ff,
    )

logger.info(f"Succesfully completed!")
