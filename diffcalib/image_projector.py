import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

# class ImageProjModel(nn.Module):
class ImageProjModel(ModelMixin, ConfigMixin):
    """Projection Model"""
    @register_to_config
    def __init__(
        self, 
        cross_attention_dim=1024, 
        clip_embeddings_dim=1024, 
        clip_extra_context_tokens=4, 
        learnable_embedding=False
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
        if learnable_embedding:
            self.learnable_embedding = nn.Embedding()


    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

    # def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
    