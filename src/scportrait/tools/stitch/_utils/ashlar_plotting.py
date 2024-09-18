import os

import matplotlib.pyplot as plt
import networkx as nx  # required for plotting results
import numpy as np

from scportrait.tools.stitch._utils.graphs import gt2nx

# try to import the graph_tool backend
try:
    from graph_tool import Graph as gtGraph
except ImportError:
    gtGraph = None

nx.graph.Graph


def draw_mosaic_image(ax, aligner, img, **kwargs):
    if img is None:
        img = [[0]]
    h, w = aligner.mosaic_shape
    ax.imshow(img, extent=(-0.5, w - 0.5, h - 0.5, -0.5), **kwargs)


def plot_edge_quality(aligner, outdir, img=None, show_tree=True, pos="metadata", im_kwargs=None, nx_kwargs=None):
    if pos == "metadata":
        centers = aligner.metadata.centers - aligner.metadata.origin
    elif pos == "aligner":
        centers = aligner.centers
    else:
        raise ValueError("pos must be either 'metadata' or 'aligner'")
    if im_kwargs is None:
        im_kwargs = {}
    if nx_kwargs is None:
        nx_kwargs = {}
    final_nx_kwargs = dict(width=2, node_size=100, font_size=6)
    final_nx_kwargs.update(nx_kwargs)
    if show_tree:
        nrows, ncols = 1, 2
        if aligner.mosaic_shape[1] * 2 / aligner.mosaic_shape[0] > 2 * 4 / 3:
            nrows, ncols = ncols, nrows
    else:
        nrows, ncols = 1, 1

    fig = plt.figure(figsize=(100, 100))
    ax = plt.subplot(nrows, ncols, 1)
    draw_mosaic_image(ax, aligner, img, **im_kwargs)
    error = np.array([aligner._cache[tuple(sorted(e))][1] for e in aligner.neighbors_graph.edges])

    # Manually center and scale data to 0-1, except infinity which is set to -1.
    # This lets us use the purple-green diverging color map to color the graph
    # edges and cause the "infinity" edges to disappear into the background
    # (which is itself purple).
    infs = error == np.inf
    error[infs] = -1
    if not infs.all():
        error_f = error[~infs]
        emin = np.min(error_f)
        emax = np.max(error_f)
        if emin == emax:
            # Always true when there's only one edge. Otherwise it's unlikely
            # but theoretically possible.
            erange = 1
        else:
            erange = emax - emin
        error[~infs] = (error_f - emin) / erange
    # Neighbor graph colored by edge alignment quality (brighter = better).
    nx.draw(
        aligner.neighbors_graph,
        ax=ax,
        with_labels=True,
        pos=np.fliplr(centers),
        edge_color=error,
        edge_vmin=-1,
        edge_vmax=1,
        edge_cmap=plt.get_cmap("PRGn"),
        **final_nx_kwargs,
    )

    if show_tree:
        ax = plt.subplot(nrows, ncols, 2)
        draw_mosaic_image(ax, aligner, img, **im_kwargs)

        # convert spanning tree to nx graph
        if isinstance(aligner.spanning_tree, nx.graph.Graph):
            spanning_tree = aligner.spanning_tree
        elif gtGraph is not None:
            if isinstance(aligner.spanning_tree, gtGraph):
                spanning_tree = gt2nx(aligner.spanning_tree)
            else:
                raise ValueError("spanning_tree must be a networkx or graph-tool graph")
        else:
            raise ImportError("graph-tool is required for this type of graph.")

        # Spanning tree with nodes at original tile positions.
        nx.draw(
            spanning_tree, ax=ax, with_labels=True, pos=np.fliplr(centers), edge_color="royalblue", **final_nx_kwargs
        )

    fig.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "QC_edge_quality.pdf"))


def plot_edge_scatter(aligner, outdir, annotate=True):
    import seaborn as sns

    xdata = aligner.all_errors

    if "cached_errors" in aligner.__dict__.keys():
        ydata = np.clip([np.linalg.norm(v[0]) for v in aligner.cached_errors.values()], 0.01, np.inf)
    else:
        ydata = np.clip([np.linalg.norm(v[0]) for v in aligner._cache.values()], 0.01, np.inf)

    # remove inf values if present
    if np.inf in ydata:
        ydata[ydata == np.inf] = np.max(ydata[ydata != np.inf]) * 2
    if np.inf in xdata:
        xdata[xdata == np.inf] = np.max(xdata[xdata != np.inf]) * 2

    pdata = np.clip(aligner.errors_negative_sampled, 0, 10)  # by clipping no inf values can remain

    g = sns.JointGrid(x=xdata, y=ydata)
    g.plot_joint(sns.scatterplot, alpha=0.5)

    _, xbins = np.histogram(np.hstack([xdata, pdata]), bins=40)

    sns.histplot(xdata, kde=False, bins=xbins, stat="density", alpha=0.4, edgecolor=(1, 1, 1, 0.4), ax=g.ax_marg_x)
    sns.histplot(pdata, bins=xbins, stat="density", alpha=0.4, element="step", fill=False, ax=g.ax_marg_x)

    g.ax_joint.axvline(aligner.max_error, c="k", ls=":")
    g.ax_joint.axhline(aligner.max_shift_pixels, c="k", ls=":")
    g.ax_joint.set_yscale("log")
    g.set_axis_labels("error", "shift")
    if annotate:
        for pair, x, y in zip(aligner.neighbors_graph.edges, xdata, ydata):
            plt.annotate(str(pair), (x, y), alpha=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "QC_edge_scatter.pdf"))
