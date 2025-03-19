try:
    import matplotlib.pyplot as plt
    import spatialdata
    import spatialdata_plot
    from spatialdata import to_polygons

except ImportError:
    raise ImportError(
        "spatialdata_plot must be installed to use the plotting capabilites. please install with `pip install spatialdata-plot`."
    ) from None

PALETTE = [
    "blue",
    "green",
    "red",
    "yellow",
    "purple",
    "orange",
    "pink",
    "cyan",
    "magenta",
    "lime",
    "teal",
    "lavender",
    "brown",
    "beige",
    "maroon",
    "mint",
    "olive",
    "apricot",
    "navy",
    "grey",
    "white",
    "black",
]


def plot_segmentation_mask(
    sdata: spatialdata.SpatialData,
    masks: list[str],
    max_width: int = 1000,
    title: str | None = None,
    select_region: tuple[int, int] | None = None,
    return_fig: bool = False,
    axs: plt.Axes | None = None,
    show_fig: bool = True,
) -> plt.Figure | None:
    # remove points object as this makes it
    points_keys = list(sdata.points.keys())
    if len(points_keys) > 0:
        for x in points_keys:
            del sdata.points[x]

    c, x, y = sdata["input_image"].scale0.image.shape
    channel_names = sdata["input_image"].scale0.c.values

    # do not plot more than 4 channels
    if c > 4:
        c = 4
    palette = PALETTE[:c]
    channel_names = list(channel_names[:c])

    # subset spatialdata object if its too large
    width = max_width // 2
    if x > max_width or y > max_width:
        if select_region is None:
            center_x = x // 2
            center_y = y // 2
        else:
            center_x, center_y = select_region

        sdata = sdata.query.bounding_box(
            axes=["x", "y"],
            min_coordinate=[center_x - width, center_y - width],
            max_coordinate=[center_x + width, center_y + width],
            target_coordinate_system="global",
        )

    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    # plot background image
    sdata.pl.render_images("input_image", channel=channel_names, palette=palette).pl.show(ax=axs, title="")

    # plot selected segmentation masks
    for mask in masks:
        assert mask in sdata, f"Mask {mask} not found in sdata object."
        sdata[f"{mask}_vectorized"] = to_polygons(sdata[mask])
        sdata.pl.render_shapes(
            f"{mask}_vectorized", fill_alpha=0, outline_alpha=0.7, outline_width=1, outline_color="white"
        ).pl.show(ax=axs, title=mask)

    # turn off axis
    axs.axis("off")
    axs.set_title(title)

    # return elements
    if return_fig:
        return fig
    elif show_fig:
        plt.show()
        return None
    else:
        return None
