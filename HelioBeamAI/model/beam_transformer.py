import torch
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from torch import nn, Tensor
from typing import Optional, Callable, Union
from HelioBeamAI.config import DEVICE

class CustomTransformerDecoderLayer(TransformerDecoderLayer):
    """
    Custom TransformerDecoderLayer without self-attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        # Initialize the parent class
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        """
        Override self-attention block to do nothing.
        """
        return torch.zeros_like(x)  # Effectively skips self-attention

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None, 
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_is_causal: bool = False,
        tgt_is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass without self-attention.
        """
        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class BeamTransformer(nn.Module):
    """
    Custom Transformer using a modified decoder without self-attention.
    """
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )

        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self.decoder_layer = CustomTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )

        self.decoder = TransformerDecoder(self.decoder_layer, num_decoder_layers)

        self.to(DEVICE)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output


if __name__ == "__main__":
    # Create the custom transformer
    model = BeamTransformer(
        d_model=64,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        batch_first=True,
    )

    # Example inputs
    src = torch.rand(32, 10, 64)  # (batch, seq, feature)
    tgt = torch.rand(32, 20, 64)  # (batch, seq, feature)

    # Forward pass
    output = model(src, tgt)
    print(output.shape)  # Output shape: (batch, tgt_seq, d_model)
