from .CBAM import CBAM
from .ECA import ECA
from .CA import CA
from .CARAFE import CARAFE
from .ASFFLite import ASFFLite
from .AAGF import AAGF
ATTN_MODELS = {
    "CBAM": CBAM,
    "ECA": ECA,
    "CA": CA,
    "CARAFE": CARAFE,
}

__all__ = ["CBAM", "ECA", "CA", "CARAFE", "ASFFLite", "AAGF"]
