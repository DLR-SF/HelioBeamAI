import os
import torch
from torch.utils.data import Dataset, DataLoader

# Define dataset paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
HELIOSAT_DATASET_PATH = os.path.join(DATA_DIR, "heliostat_dataset.pt")
ARTIFICIAL_DATASET_PATH = os.path.join(DATA_DIR, "artificial_dataset.pt")

class HeliostatDataset(Dataset):
    def __init__(self, data, images, artificial_data=None, artificial_images=None, mode="train", encoder_size=20, decoder_size=20,artifical_data_weight = 1.0):
        """
        Args:
            data (dict): Real numerical data {heliostat_id: (train_samples, test_samples)}
            images (dict): Real image data {heliostat_id: (train_images, test_images)}
            artificial_data (dict, optional): Artificial numerical data {heliostat_id: (encoder_samples, decoder_samples)}
            artificial_images (dict, optional): Artificial image data {heliostat_id: (encoder_images, decoder_images)}
            mode (str): "train" or "valid" (determines how we split samples).
            encoder_size (int): Fixed number of samples for encoder.
            decoder_size (int): Fixed number of samples for decoder.
        """
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size

        def ensure_tensor(tensor):
            """Ensure input is a tensor without unnecessary conversions."""
            return tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor, dtype=torch.float32)

        self.data = {
            k: (
                ensure_tensor(v[0]),  # Only convert if not a tensor
                ensure_tensor(v[1])[:decoder_size]  # Truncate test samples
            )
            for k, v in data.items()
        }

        self.images = {
            k: (
                ensure_tensor(v[0]),
                ensure_tensor(v[1])[:decoder_size]  # Truncate test images
            )
            for k, v in images.items()
        }

        self.artificial_data = (
            {
                k: (
                    ensure_tensor(v[0])[:encoder_size],  # Truncate training samples
                    ensure_tensor(v[1])[:decoder_size]   # Truncate test samples
                )
                for k, v in artificial_data.items()
            }
            if artificial_data else {}
        )

        self.artificial_images = (
            {
                k: (
                    ensure_tensor(v[0])[:encoder_size],  # Truncate training images
                    ensure_tensor(v[1])[:decoder_size]   # Truncate test images
                )
                for k, v in artificial_images.items()
            }
            if artificial_images else {}
        )

        self.mode = mode

        self.heliostats = list(self.data.keys())  # List of real heliostat IDs
        self.artificial_heliostats = list(self.artificial_data.keys()) if self.artificial_data else []

        # Validate data constraints
        self._validate_real_data()
        self._validate_artifical_data()
        
        
        # Reweighting
        artifical_data_weight = 1.0
        self.artifical_data_multiplier = artifical_data_weight * (len(self.heliostats)/len(self.artificial_heliostats))
        print(f'Multiplier: {self.artifical_data_multiplier}')

    def _validate_real_data(self):
        """Check if real data sequences meet encoder and decoder size requirements."""
        for heliostat_id, (train_samples, test_samples) in self.artificial_data.items():
            if len(test_samples) < self.decoder_size:
                raise ValueError(f"Heliostat {heliostat_id} has only {len(test_samples)} test samples, but needs at least {self.decoder_size}.")
            if len(train_samples) < (self.encoder_size):
                raise ValueError(f"Heliostat {heliostat_id} has only {len(train_samples)} train samples, but needs at least {self.encoder_size + self.decoder_size}.")

    def _validate_artifical_data(self):
        """Check if real data sequences meet encoder and decoder size requirements."""
        for heliostat_id, (train_samples, test_samples) in self.data.items():
            if len(test_samples) < self.decoder_size:
                raise ValueError(f"Heliostat {heliostat_id} has only {len(test_samples)} test samples, but needs at least {self.decoder_size}.")
            if len(train_samples) < (self.encoder_size + self.decoder_size):
                raise ValueError(f"Heliostat {heliostat_id} has only {len(train_samples)} train samples, but needs at least {self.encoder_size}.")
            
    def __len__(self):
        """Returns total number of heliostats (real + artificial if available)."""
        return len(self.heliostats) + int(len(self.artificial_heliostats)*self.artifical_data_multiplier)

    def __getitem__(self, idx):
        """
        Returns a single sample containing:
        - Encoder & decoder inputs for **real or artificial** heliostat data.
        - Corresponding **image sequences**.
        """
        if idx < len(self.heliostats):
            # ===== REAL DATA (Random Sampling) =====
            heliostat_id = self.heliostats[idx]
            train_samples, test_samples = self.data[heliostat_id]
            train_images, test_images = self.images[heliostat_id]

            if self.mode == "train":
                # Randomly sample 25 encoder + 15 decoder from training data
                indices = torch.randperm(train_samples.shape[0])[: self.encoder_size + self.decoder_size]
                encoder_idx = indices[: self.encoder_size]
                decoder_idx = indices[self.encoder_size:]

                encoder_samples = train_samples[encoder_idx]
                decoder_samples = train_samples[decoder_idx]

                encoder_images = train_images[encoder_idx]
                decoder_images = train_images[decoder_idx]
            else:
                # Valid mode: Use first encoder_size training samples and first decoder_size test samples
                encoder_samples = train_samples[: self.encoder_size]
                decoder_samples = test_samples[: self.decoder_size]
                encoder_images = train_images[: self.encoder_size]
                decoder_images = test_images[: self.decoder_size]

        else:
            # ===== ARTIFICIAL DATA (Fixed Sequences) =====
            artificial_idx = (idx - len(self.heliostats))%len(self.artificial_heliostats)
            heliostat_id = self.artificial_heliostats[artificial_idx]

            encoder_samples, decoder_samples = self.artificial_data[heliostat_id]
            encoder_images, decoder_images = self.artificial_images[heliostat_id]

        return {
            "encoder_input": encoder_samples,  # Shape: [20, 6]
            "decoder_input": decoder_samples,  # Shape: [20, 6]
            "encoder_images": encoder_images,  # Shape: [20, 64, 64]
            "decoder_images": decoder_images,  # Shape: [20, 64, 64]
            "is_artificial": torch.tensor(1 if idx >= len(self.heliostats) else 0, dtype=torch.int64)
        }

def load_datasets():
    """Loads dataset files from the data directory."""
    print(f"🔍 Loading datasets from {DATA_DIR}...")

    if not os.path.exists(HELIOSAT_DATASET_PATH):
        raise FileNotFoundError(f"❌ ERROR: Missing dataset file: {HELIOSAT_DATASET_PATH}")

    heliostat_data = torch.load(HELIOSAT_DATASET_PATH)
    real_data = heliostat_data.get("real_data")
    real_images = heliostat_data.get("real_images")

    artificial_data, artificial_images = None, None
    if os.path.exists(ARTIFICIAL_DATASET_PATH):
        artificial_data_dict = torch.load(ARTIFICIAL_DATASET_PATH)
        artificial_data = artificial_data_dict.get("artificial_data")
        artificial_images = artificial_data_dict.get("artificial_images")
        print("✅ Artificial dataset loaded successfully.")

    print("✅ Heliostat dataset loaded successfully.")
    return real_data, real_images, artificial_data, artificial_images

def get_dataloaders(batch_size=16, num_workers=4):
    """
    Loads datasets from the 'data/' directory and creates training & validation dataloaders.

    Args:
        batch_size (int): Number of heliostats per batch.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        train_loader, val_loader: PyTorch DataLoaders for training and validation.
    """
    real_data, real_images, artificial_data, artificial_images = load_datasets()
    
    train_dataset = HeliostatDataset(real_data, real_images, artificial_data, artificial_images, mode="train")
    val_dataset = HeliostatDataset(real_data, real_images, artificial_data, artificial_images, mode="valid")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    return train_loader, val_loader