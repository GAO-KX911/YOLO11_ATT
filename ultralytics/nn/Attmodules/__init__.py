from .CBAM import CBAM
from .ECA import ECA
ATTN_MODELS = {
    "CBAM": CBAM,
    "ECA": ECA
}

__all__ = ["CBAM", "ECA"]