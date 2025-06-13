# utils/config_loader.py
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml" # Assumes config.yaml is in root

def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config

# You can also add functions to resolve paths directly if you like
def get_model_path(model_name_key: str):
    config = load_config()
    return Path(config['data_paths']['models_dir']) / config['data_paths'][model_name_key]

def get_preprocessed_data_path(data_file_key: str):
    config = load_config()
    return Path(config['data_paths']['preprocessed_dir']) / config['data_paths'][data_file_key]