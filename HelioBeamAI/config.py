import os

# General settings
PROJECT_NAME = "HelioBeamAI"
DEVICE = "cuda:1"
SEED = 42  # For reproducibility

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODEL_DIR = os.path.join(BASE_DIR, "../models")

HELIOSAT_DATASET_PATH = os.path.join(DATA_DIR, "heliostat_dataset.pt")
ARTIFICIAL_DATASET_PATH = os.path.join(DATA_DIR, "artificial_dataset.pt")

# Transformer Model Hyperparameters
TRANSFORMER_CONFIG = {
    "d_model": 64,
    "ffn_hidden": 64,  # Same as d_model
    "n_head": 4,
    "max_len": 512,  # Maximum sequence length
    "n_layers_enc": 2,
    "n_layers_dec": 4,
    "drop_prob": 0.1,
    "init_lr": 1e-5,
    "weight_decay": 1e-8,
    "batch_size": 16,
}

# ⚙ VAE Configuration
VAE_CONFIG = {
    "ddconfig": {
        "ch": 4,
        "out_ch": 1,
        "ch_mult": (1, 2, 4, 8),
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.2,
        "resamp_with_conv": True,
        "in_channels": 1,
        "resolution": 64,
        "z_channels": 1,
        "double_z": True,
        "use_linear_attn": False,
        "attn_type": "vanilla",
        "give_pre_end": False,
        "tanh_out": False,
    },
    "lossconfig": {
        "target": "torch.nn.Identity",
        "params": {
            "disc_start": 50001,
            "kl_weight": 1.0e-06,
            "disc_weight": 0.5,
        },
    },
    "embed_dim": 4,
}


# Function to print all configurations
def print_config():
    print("📌 Configuration for HelioBeamAI:")
    print(f"Device: {DEVICE}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print("\n🔹 Transformer Config:")
    for k, v in TRANSFORMER_CONFIG.items():
        print(f"  {k}: {v}")
    print("\n🔹 CNN Config:")
    for k, v in VAE_CONFIG.items():
        print(f"  {k}: {v}")

# Run this script directly to print the config
if __name__ == "__main__":
    print_config()
