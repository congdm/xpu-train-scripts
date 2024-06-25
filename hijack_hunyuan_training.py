import torch
import intel_extension_for_pytorch as ipex

from diffusers.models.transformers import HunyuanDiT2DModel
from HunyuanDiT.hydit.diffusion import gaussian_diffusion

from config import args

###################################################################################################
# hydit.diffusion.gaussian_diffusion patching
def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    #res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    res = torch.from_numpy(arr).float().to(device=timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
gaussian_diffusion._extract_into_tensor = extract_into_tensor

def gaussian_diffusion_q_sample(self, x_start, t, noise=None):
    """
    Diffuse the data for a given number of diffusion steps.

    In other words, sample from q(x_t | x_0).

    :param x_start: the initial data batch.
    :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    :param noise: if specified, the split-out normal noise.
    :return: A noisy version of x_start.
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    gaussian_diffusion.assert_shape(noise, x_start)
    org_dtype = x_start.dtype
    return (
        gaussian_diffusion._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + gaussian_diffusion._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        * noise
    ).to(dtype=org_dtype)
gaussian_diffusion.GaussianDiffusion.q_sample = gaussian_diffusion_q_sample

##################################################################################################

# diffusers.models.transformers.HunyuanDiT2DModel patching
# manually add gradient checkpointing
def HunyuanDiT2DModel_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states=None,
    text_embedding_mask=None,
    encoder_hidden_states_t5=None,
    text_embedding_mask_t5=None,
    image_meta_size=None,
    style=None,
    image_rotary_emb=None,
    return_dict=True,
):
    """
    The [`HunyuanDiT2DModel`] forward method.

    Args:
    hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
        The input tensor.
    timestep ( `torch.LongTensor`, *optional*):
        Used to indicate denoising step.
    encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
        Conditional embeddings for cross attention layer. This is the output of `BertModel`.
    text_embedding_mask: torch.Tensor
        An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
        of `BertModel`.
    encoder_hidden_states_t5 ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
        Conditional embeddings for cross attention layer. This is the output of T5 Text Encoder.
    text_embedding_mask_t5: torch.Tensor
        An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
        of T5 Text Encoder.
    image_meta_size (torch.Tensor):
        Conditional embedding indicate the image sizes
    style: torch.Tensor:
        Conditional embedding indicate the style
    image_rotary_emb (`torch.Tensor`):
        The image rotary embeddings to apply on query and key tensors during attention calculation.
    return_dict: bool
        Whether to return a dictionary.
    """

    #hidden_states = hidden_states.to(dtype=torch.float16)
    #timestep = timestep.to(dtype=torch.float16)
    
    # if args.gradient_checkpointing:
    #     for x in hidden_states:
    #         x.requires_grad = True
    #     for x in encoder_hidden_states_t5:
    #         x.requires_grad = True
    #     for x in encoder_hidden_states:
    #         x.requires_grad = True
    #     for x in image_meta_size:
    #        x.requires_grad = True
    #     timestep.requires_grad = True

    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)

    temb = self.time_extra_emb(
        timestep, encoder_hidden_states_t5, image_meta_size, style, hidden_dtype=timestep.dtype
    )  # [B, D]

    # text projection
    batch_size, sequence_length, _ = encoder_hidden_states_t5.shape
    encoder_hidden_states_t5 = self.text_embedder(
        encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
    )
    encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, sequence_length, -1)

    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1)
    text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
    text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()

    encoder_hidden_states = torch.where(text_embedding_mask, encoder_hidden_states, self.text_embedding_padding)

    skips = []
    for layer, block in enumerate(self.blocks):
        if layer > self.config.num_layers // 2:
            skip = skips.pop()
            if not args.gradient_checkpointing:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                )  # (N, L, D)
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states, encoder_hidden_states, temb, image_rotary_emb, skip,
                    use_reentrant=False,
                )
        else:
            if not args.gradient_checkpointing:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )  # (N, L, D)
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                    use_reentrant=False,
                )

        if layer < (self.config.num_layers // 2 - 1):
            skips.append(hidden_states)

    # final layer
    hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
    hidden_states = self.proj_out(hidden_states)
    # (N, L, patch_size ** 2 * out_channels)

    # unpatchify: (N, out_channels, H, W)
    patch_size = self.pos_embed.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )
    return {'x': output.float()}

HunyuanDiT2DModel.forward = HunyuanDiT2DModel_forward

