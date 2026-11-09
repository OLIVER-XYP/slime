"""Qwen3 bridge extensions for the dependency-free Megatron local backend."""

from mbridge.core import register_model
from mbridge.models import Qwen3Bridge


@register_model("qwen3")
class Qwen3LocalBridge(Qwen3Bridge):
    """Map standalone local-backend LayerNorm modules to Qwen3 HF weights."""

    _MLP_MAPPING = Qwen3Bridge._MLP_MAPPING | {
        "pre_mlp_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
    }
    _OTHER_MAPPING = {
        "input_layernorm.weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
    }
