# silence warnings
import warnings

# silence warning from cellpose resulting in missing parameter set in model call see #141
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

# silence warning from huggingface transformers package
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated and will be removed in version 1.0.0.*",
    category=FutureWarning,
)
