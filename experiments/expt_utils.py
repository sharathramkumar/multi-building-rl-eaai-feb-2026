import yaml, os
from datetime import datetime, timedelta

next_day = lambda d: (datetime.strptime(d, "%m/%d/%Y") + timedelta(days=1)).strftime(
    "%m/%d/%Y"
)


def load_configs(DATA_CONFIG_PATH):
    with open(DATA_CONFIG_PATH) as f:
        data_cfg = yaml.safe_load(f)
        data_root_path = os.path.abspath(
            os.path.join(os.path.dirname(DATA_CONFIG_PATH), data_cfg["data_root"])
        )
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    return data_cfg, data_root_path, cfg


def find_cons_models_path(data_root_path, data_cfg, cfg, is_train=False):
    inner_cfg_key = "consumer_models" if not is_train else "train_consumer_models"
    return os.path.join(
        data_root_path,
        data_cfg["consumer_models"][cfg[inner_cfg_key]["type"]][
            cfg[inner_cfg_key]["override"]
        ],
    )
