import pytest
import torch
from torch.utils.data import DataLoader
from HelioBeamAI.dataset import HeliostatDataset, get_dataloaders  # Import your dataset and dataloader function

# Create mock data
@pytest.fixture
def mock_data():
    heliostat_ids = ["H1", "H2", "H3"]  # Example heliostat names

    real_data = {
        h: (
            torch.randn(50, 6),  # 50 training samples (random values)
            torch.randn(25, 6)   # 20 test samples (should be truncated to 15)
        )
        for h in heliostat_ids
    }

    real_images = {
        h: (
            torch.randn(50, 64, 64),  # 50 training images
            torch.randn(25, 64, 64)   # 20 test images (should be truncated to 15)
        )
        for h in heliostat_ids
    }

    artificial_data = {
        "A1": (torch.randn(20, 6), torch.randn(20, 6))
    }
    
    artificial_images = {
        "A1": (torch.randn(20, 64, 64), torch.randn(20, 64, 64))
    }

    return real_data, real_images, artificial_data, artificial_images


def test_dataset_initialization(mock_data):
    """Test dataset creation and length calculation."""
    real_data, real_images, artificial_data, artificial_images = mock_data

    dataset = HeliostatDataset(real_data, real_images, artificial_data, artificial_images, mode="train")

    # Check dataset length (3 real + 1 artificial)
    assert len(dataset) == 4, f"Expected dataset length 4, but got {len(dataset)}"

    # Check first sample structure
    sample = dataset[0]
    assert sample["encoder_input"].shape == (20, 6)
    assert sample["decoder_input"].shape == (20, 6)
    assert sample["encoder_images"].shape == (20, 64, 64)
    assert sample["decoder_images"].shape == (20, 64, 64)
    assert sample["is_artificial"].dtype == torch.int64


def test_dataloader(mock_data):
    """Test that DataLoader works correctly."""
    real_data, real_images, artificial_data, artificial_images = mock_data

    train_loader, valid_loader = get_dataloaders(real_data, real_images, artificial_data, artificial_images, batch_size=2)

    # Check that batches load correctly
    for batch in train_loader:
        assert batch["encoder_input"].shape == (2, 20, 6)  # Batch size 2, 25 encoder samples, 6 features
        assert batch["decoder_input"].shape == (2, 20, 6)  # Batch size 2, 15 decoder samples, 6 features
        assert batch["encoder_images"].shape == (2, 20, 64, 64)
        assert batch["decoder_images"].shape == (2, 20, 64, 64)
        assert batch["is_artificial"].shape == (2,)

        break  # Only test one batch to avoid unnecessary computations


def test_dataset_constraints(mock_data):
    """Ensure dataset validation rejects incorrect data sizes."""
    real_data, real_images, artificial_data, artificial_images = mock_data

    # Modify real_data to fail validation
    real_data["H1"] = (torch.randn(30, 6), torch.randn(10, 6))  # Not enough train/test samples

    with pytest.raises(ValueError, match="Heliostat H1 has only 10 test samples, but needs at least 20."):
        HeliostatDataset(real_data, real_images, artificial_data, artificial_images, mode="train")


def test_artificial_data(mock_data):
    """Ensure artificial data is correctly loaded."""
    real_data, real_images, artificial_data, artificial_images = mock_data

    dataset = HeliostatDataset(real_data, real_images, artificial_data, artificial_images, mode="train")

    artificial_sample = dataset[len(real_data)]  # First artificial sample
    assert artificial_sample["is_artificial"] == 1, "Artificial sample should have is_artificial = 1"

    assert artificial_sample["encoder_input"].shape == (20, 6)
    assert artificial_sample["decoder_input"].shape == (20, 6)
    assert artificial_sample["encoder_images"].shape == (20, 64, 64)
    assert artificial_sample["decoder_images"].shape == (20, 64, 64)
