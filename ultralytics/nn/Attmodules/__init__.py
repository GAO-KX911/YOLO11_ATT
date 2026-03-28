from .CBAM import CBAM
from .ECA import ECA
from.CA import CA
ATTN_MODELS = {
    "CBAM": CBAM,
    "ECA": ECA,
    "CA": CA
}

__all__ = ["CBAM", "ECA"]