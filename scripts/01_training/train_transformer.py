import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from HelioBeamAI.model.transformer_module import TransformerModule
from HelioBeamAI.config import DEVICE
from HelioBeamAI.dataset import get_dataloaders  # Your data loader

torch.set_float32_matmul_precision('medium') 

# Set up dataloaders
train_loader, val_loader = get_dataloaders()

# Initialize model
model = TransformerModule(vae_state_dict='vae_state_dict')

# Callbacks
checkpoint_callback = ModelCheckpoint(monitor="loss_val_ep", save_top_k=3, mode="min")

# Ensure DEVICE is set correctly
DEVICE = DEVICE.lower().strip()  # Normalize input (avoid casing issues)

# Handle CUDA selection
if DEVICE.startswith("cuda"):
    accelerator = "gpu"

    # Extract device index if specified (e.g., cuda:0 or cuda:2)
    if ":" in DEVICE:
        devices = [int(DEVICE.split(":")[1])]  # Extract the GPU index
    else:
        devices = "auto"  # Use all available GPUs
else:
    accelerator = "cpu"
    devices = "auto"  # Use all CPU cores

print(f"🚀 Using {accelerator} on devices: {devices}")

# Create PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=100000,
    accelerator=accelerator, 
    devices=devices,    
    callbacks=[checkpoint_callback],
    log_every_n_steps=150
)

# Train the model
trainer.fit(model, train_loader, val_loader)
