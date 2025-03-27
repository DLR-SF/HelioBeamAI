import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.optim
from adamp import AdamP
from torch.optim.lr_scheduler import ReduceLROnPlateau

from HelioBeamAI.config import TRANSFORMER_CONFIG
from HelioBeamAI.model.beam_transformer import BeamTransformer
from HelioBeamAI.utils import calcFluxDev, loadVAE, generate_image_grid

class PositionwiseFeedForward(torch.nn.Module):

    def __init__(self, inp, out, drop_prob=0.0):
        super(PositionwiseFeedForward, self).__init__()
        
        hidden = int(max(inp,out)/2)
        self.linear1 = torch.nn.Linear(inp, hidden)
        self.linear2 = torch.nn.Linear(hidden, out)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class TransformerModule(pl.LightningModule):
    def __init__(self, init_lr=1e-5, 
                 weight_decay=1e-8, 
                 state_dict_path=None, 
                 vae_state_dict=None,
                ):
        super().__init__()
        
        # Transformer model
        self.model = BeamTransformer(
            d_model=TRANSFORMER_CONFIG["d_model"],
            dim_feedforward=TRANSFORMER_CONFIG["ffn_hidden"],
            nhead=TRANSFORMER_CONFIG["n_head"],
            num_encoder_layers=TRANSFORMER_CONFIG["n_layers_enc"],
            num_decoder_layers=TRANSFORMER_CONFIG["n_layers_dec"],
            dropout=TRANSFORMER_CONFIG["drop_prob"],
            device=None,
        )
        
        # Load VAE model
        self.vae = loadVAE(vae_state_dict)
        self.z_shape = self.vae.decoder.z_shape[1:]
        
        # Loss settings
        self.scale_latent_loss = 1e-1
        self.criterion_latent = torch.nn.MSELoss()

        # Load state dict if provided
        if state_dict_path is not None:
            checkpoint = torch.load(state_dict_path, map_location=self.device)
            model_state_dict = checkpoint['state_dict']
            new_state_dict = {k.replace('model.', ''): v for k, v in model_state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)

        # Hyperparameters
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.d_model = TRANSFORMER_CONFIG["d_model"]
        
        self.enc_emb = PositionwiseFeedForward(inp=71,out=self.d_model)
        self.dec_emb = PositionwiseFeedForward(inp=7,out=self.d_model)

    def forward(self, imgs, src, trg):
        
        latent = self.calcLatentFromFlux(imgs)
        src = torch.cat([latent, src], dim=-1)
        src_emb = self.enc_emb(src)
        trg_emb = self.dec_emb(trg)
        output = self.model(src_emb, trg_emb)
        flux = self.calcFluxFromLatent(output)
        
        return flux
    
    def training_step(self, batch, batch_idx):
        """
        Training step:
        """
        src = batch['encoder_input']
        trg = batch['decoder_input']
        imgs = batch['encoder_images']
        imgs_gt = batch['decoder_images']
        
        flux = self.forward(imgs,src,trg)
        loss_flux = calcFluxDev(imgs_gt, flux)
        loss = torch.mean(loss_flux)

        self.log('loss_flux', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step:
        """
        # Test set loss
        src = batch['encoder_input']
        trg = batch['decoder_input']
        imgs = batch['encoder_images']
        imgs_gt = batch['decoder_images']
        
        flux = self.forward(imgs,src,trg)
        loss_val = torch.mean(calcFluxDev(imgs_gt, flux))

        self.log('loss_val_ep', loss_val, on_epoch=True)
        
        # Log images every 100 epochs
        if batch_idx == 0 and self.current_epoch % 100 == 0:
            img = generate_image_grid(imgs_gt, flux)
            self.logger.experiment.add_image('predictions_val', img, global_step=self.current_epoch)

        return loss_val

    def configure_optimizers(self):
        """
        Configures AdamP optimizer with ReduceLROnPlateau scheduler.
        """
        optimizer = AdamP(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        return optimizer
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss_val_ep",  # Use validation loss to adjust LR
                "interval": "epoch",
                "frequency": 1
            }
        }

    def calcFluxFromLatent(self, latent):
        """
        Converts latent space representation to flux using VAE decoder.
        """
        b, s, d = latent.size()
        latent = latent.view(b * s, d)
        batch_z_shape = (latent.size(0), *self.z_shape)
        flux = self.vae.decoder(latent.view(*batch_z_shape))
        return flux.view(b, s, flux.size(-2), flux.size(-1))

    def calcLatentFromFlux(self, imgs):
        """
        Encodes images into latent space using VAE.
        """
        b, s, d, _ = imgs.size()
        imgs = imgs.view(b * s, 1, d, d)
        post = self.vae.encode(imgs).mode()
        post = self.vae.post_quant_conv(post)
        return post.view(b, s, -1)