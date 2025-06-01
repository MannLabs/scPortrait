from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError:
    raise ImportError(
        "matplotlib_scalebar must be installed to use the plotting capabilities. please install with `pip install 'scportrait[plotting]'`."
    ) from None


def _custom_cmap():
    # Define the colors: 0 is transparent, 1 is red, 2 is blue
    colors = [
        (0, 0, 0, 0),  # Transparent
        (1, 0, 0, 0.4),  # Red
        (0, 0, 1, 0.4),
    ]  # Blue

    # Create the colormap
    cmap = ListedColormap(colors)

    # Define the boundaries and normalization
    bounds = [0, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    return (cmap, norm)


def add_scalebar(
    ax: Axes,
    resolution: float,
    resolution_unit: str = "um",
    fixed_length: float | None = None,
    location: str = "lower right",
    color: str = "white",
    scale_loc: str = "bottom",
    border_pad=1,
) -> None:
    """Add a scalebar to an axis.

    This function wraps the package `matplotlib-scalebar <https://github.com/ppinard/matplotlib-scalebar>`_. Please utilize the original package if you require more customization.

    Args:
        ax: The axis to add the scalebar to.
        resolution: The resolution of the image.
        resolution_unit: The unit of the resolution.
        fixed_length: specify the absolute length of the scale bar
        location: The location of the scalebar.
        color: The color of the scalebar.
        scale_location: where the text labelling the scale par should be located relative to the bar. If set to "none" then there will be no label.
        border_pad: distance between scalebar element and border of the image.
    """
    scalebar = ScaleBar(
        resolution,
        resolution_unit,
        length_fraction=0.2,
        location=location,
        frameon=False,
        color=color,
        fixed_value=fixed_length,
        scale_loc=scale_loc,
        border_pad=border_pad,
    )
    ax.add_artist(scalebar)
