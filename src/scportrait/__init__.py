# silence warnings
import warnings

# silence warning from cellpose resulting in missing parameter set in model call see #141
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)
