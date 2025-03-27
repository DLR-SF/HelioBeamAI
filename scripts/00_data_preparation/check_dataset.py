import torch
import os
from HelioBeamAI.config import HELIOSAT_DATASET_PATH, ARTIFICIAL_DATASET_PATH

# Expected keys in each dataset
EXPECTED_KEYS = {
    "heliostat_dataset": ["real_data", "real_images"],
    "artificial_dataset": ["artificial_data", "artificial_images"],
}

# Expected tensor dimensions
EXPECTED_DIMENSIONS = {
    "real_data": 2,       # Numerical data (heliostat features)
    "real_images": 3,     # Image data (batch, height, width, channels)
    "artificial_data": 2,
    "artificial_images": 3,
}


def load_dataset(file_path, dataset_name):
    """Loads a dataset file and checks for required keys."""
    if not os.path.exists(file_path):
        print(f"❌ ERROR: {dataset_name} file not found at {file_path}")
        return None

    try:
        dataset = torch.load(file_path)
    except Exception as e:
        print(f"❌ ERROR: Could not load {dataset_name}: {e}")
        return None

    # Validate required keys
    missing_keys = [key for key in EXPECTED_KEYS[dataset_name] if key not in dataset]
    if missing_keys:
        print(f"❌ ERROR: {dataset_name} is missing keys: {missing_keys}")
        return None

    return dataset


def check_tensor_structure(data, key_name, expected_dim):
    """Checks if all entries in a dataset dictionary have valid tensors with expected dimensions."""
    if not isinstance(data, dict):
        print(f"❌ ERROR: {key_name} should be a dictionary but is {type(data)}.")
        return False

    for heliostat_id, (train, test) in data.items():
        if not isinstance(train, torch.Tensor) or not isinstance(test, torch.Tensor):
            print(f"❌ ERROR: {key_name} - Heliostat {heliostat_id} contains non-tensor data.")
            return False

        if train.dim() != expected_dim or test.dim() != expected_dim:
            print(f"❌ ERROR: {key_name} - Heliostat {heliostat_id} has incorrect tensor dimensions.")
            return False

    return True


def check_dataset_integrity():
    """Loads and validates both heliostat and artificial datasets."""
    print("🔍 Checking dataset files in HelioBeamAI/data/\n")

    # Check Heliostat Dataset
    heliostat_data = load_dataset(HELIOSAT_DATASET_PATH, "heliostat_dataset")
    if heliostat_data:
        print("📂 Checking heliostat_dataset.pt...")

        if check_tensor_structure(heliostat_data["real_data"], "real_data", EXPECTED_DIMENSIONS["real_data"]) and \
           check_tensor_structure(heliostat_data["real_images"], "real_images", EXPECTED_DIMENSIONS["real_images"]):
            print("✅ Heliostat dataset is valid!\n")

    # Check Artificial Dataset
    artificial_data = load_dataset(ARTIFICIAL_DATASET_PATH, "artificial_dataset")
    if artificial_data:
        print("📂 Checking artificial_dataset.pt...")

        if check_tensor_structure(artificial_data["artificial_data"], "artificial_data", EXPECTED_DIMENSIONS["artificial_data"]) and \
           check_tensor_structure(artificial_data["artificial_images"], "artificial_images", EXPECTED_DIMENSIONS["artificial_images"]):
            print("✅ Artificial dataset is valid!\n")

    print("🎉 All datasets have been verified and are correctly structured!")


if __name__ == "__main__":
    check_dataset_integrity()
