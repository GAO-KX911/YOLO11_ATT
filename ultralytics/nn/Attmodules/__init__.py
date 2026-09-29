from .AAGF import AAGF
from .ASFFLite import ASFFLite
from .CA import CA
from .CARAFE import CARAFE
from .CBAM import CBAM
from .ECA import ECA
from .P2DRFG import P2DRFG

ATTN_MODELS = {
    "CBAM": CBAM,
    "ECA": ECA,
    "CA": CA,
    "CARAFE": CARAFE,
}

__all__ = ["AAGF", "CA", "CARAFE", "CBAM", "ECA", "P2DRFG", "ASFFLite"]
