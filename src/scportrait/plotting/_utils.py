from matplotlib.colors import ListedColormap, BoundaryNorm

def _custom_cmap():

    # Define the colors: 0 is transparent, 1 is red, 2 is blue
    colors = [(0, 0, 0, 0),   # Transparent
             (1, 0, 0, 0.4),  # Red
             (0, 0, 1, 0.4)]  # Blue 

    # Create the colormap
    cmap = ListedColormap(colors)

    # Define the boundaries and normalization
    bounds = [0, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    return(cmap, norm)
