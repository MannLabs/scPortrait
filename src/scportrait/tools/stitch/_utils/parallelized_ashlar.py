"""
functions adapted from https://github.com/labsyspharm/ashlar to provide support for multi-threaded execution.
"""

import copy as copy
import sys

import numpy as np

# Import your utils module here
import sklearn.linear_model

# import graph algorithms
try:
    from graph_tool import Graph as gtGraph
    from graph_tool import GraphView
    from graph_tool.generation import remove_parallel_edges
    from graph_tool.search import bfs_iterator
    from graph_tool.topology import label_components, shortest_path

    flavor = "graph-tool"
except ImportError:
    flavor = "networkx"

import networkx as nx
from alphabase.io.tempmmap import mmap_array_from_path
from ashlar import utils as utils
from ashlar.reg import EdgeAligner, Mosaic, warn_data
from networkx import Graph as nxGraph
from tqdm.auto import tqdm

from scportrait.tools.stitch._utils.graphs import get_center_nodes, nx2gt
from scportrait.tools.stitch._utils.parallelilzation import execute_indexed_parallel, execute_parallel


class ParallelEdgeAligner(EdgeAligner):
    def __init__(
        self,
        reader,
        n_threads=20,
        channel=0,
        max_shift=15,
        alpha=0.01,
        max_error=None,
        randomize=False,
        filter_sigma=0.0,
        do_make_thumbnail=True,
        verbose=False,
    ):
        super().__init__(
            reader=reader,
            channel=channel,
            max_shift=max_shift,
            alpha=alpha,
            max_error=max_error,
            randomize=randomize,
            filter_sigma=filter_sigma,
            do_make_thumbnail=do_make_thumbnail,
            verbose=verbose,
        )

        self.n_threads = n_threads

        # set flavor to graph-tool if available
        try:
            from graph_tool import Graph as gtGraph

            self.flavor = "graph-tool"
        except ImportError:
            Warning(
                "graph-tool not available, using networkx as default. \n For stitching large datasets, graph-tool is recommended as it provides better performance."
            )
            self.flavor = "networkx"

    def compute_threshold(self):
        # Compute error threshold for rejecting aligments. We generate a
        # distribution of error scores for many known non-overlapping image
        # regions and take a certain percentile as the maximum allowable error.
        # The percentile becomes our accepted false-positive ratio.
        edges = self.neighbors_graph.edges
        num_tiles = self.metadata.num_images
        # If not enough tiles overlap to matter, skip this whole thing.
        if len(edges) <= 1:
            self.errors_negative_sampled = np.empty(0)
            self.max_error = np.inf
            return
        widths = np.array([self.intersection(t1, t2).shape.min() for t1, t2 in edges])
        w = widths.max()
        max_offset = self.metadata.size[0] - w
        # Number of possible pairs minus number of actual neighbor pairs.
        num_distant_pairs = num_tiles * (num_tiles - 1) // 2 - len(edges)
        # Reduce permutation count for small datasets -- there are fewer
        # possible truly distinct strips with fewer tiles. The calculation here
        # is just a heuristic, not rigorously derived.
        n = 1000 if num_distant_pairs > 8 else (num_distant_pairs + 1) * 10
        pairs = np.empty((n, 2), dtype=int)
        offsets = np.empty((n, 2), dtype=int)
        # Generate n random non-overlapping image strips. Strips are always
        # horizontal, across the entire image width.
        max_tries = 100
        if self.randomize is False:
            random_state = np.random.RandomState(0)
        else:
            random_state = np.random.RandomState()
        for i in range(n):
            # Limit tries to avoid infinite loop in pathological cases.
            for _current_try in range(max_tries):
                t1, t2 = random_state.randint(self.metadata.num_images, size=2)
                o1, o2 = random_state.randint(max_offset, size=2)
                # Check for non-overlapping strips and abort the retry loop.
                if t1 != t2 and (t1, t2) not in edges:
                    # Different, non-neighboring tiles -- always OK.
                    break
                elif t1 == t2 and abs(o1 - o2) > w:
                    # Same tile OK if strips don't overlap within the image.
                    break
                elif (t1, t2) in edges:
                    # Neighbors OK if either strip is entirely outside the
                    # expected overlap region (based on nominal positions).
                    its = self.intersection(t1, t2, np.repeat(w, 2))
                    ioff1, ioff2 = its.offsets[:, 0]
                    if (
                        its.shape[0] > its.shape[1]
                        or o1 < ioff1 - w
                        or o1 > ioff1 + w
                        or o2 < ioff2 - w
                        or o2 > ioff2 + w
                    ):
                        break
            else:
                # Retries exhausted. This should be very rare.
                warn_data("Could not find non-overlapping strips in {max_tries} tries")
            pairs[i] = t1, t2
            offsets[i] = o1, o2

        def register(t1, t2, offset1, offset2):
            img1 = self.reader.read(t1, self.channel)[offset1 : offset1 + w, :]
            img2 = self.reader.read(t2, self.channel)[offset2 : offset2 + w, :]
            _, error = utils.register(img1, img2, self.filter_sigma, upsample=1)
            return error

        # prepare arguments for executor
        args = []
        for (t1, t2), (offset1, offset2) in zip(pairs, offsets, strict=False):
            arg = (t1, t2, offset1, offset2)
            args.append(copy.deepcopy(arg))

        errors = execute_indexed_parallel(
            register,
            args=args,
            tqdm_args={
                "file": sys.stdout,
                "disable": not self.verbose,
                "desc": "    quantifying alignment error",
            },
            n_threads=self.n_threads,
        )

        errors = np.array(errors)
        self.errors_negative_sampled = errors
        self.max_error = np.percentile(errors, self.alpha * 100)

    def register_all(self):
        args = []
        for t1, t2 in self.neighbors_graph.edges:
            arg = (t1, t2)
            args.append(copy.deepcopy(arg))

        execute_parallel(
            self.register_pair,
            args=args,
            tqdm_args={
                "file": sys.stdout,
                "disable": not self.verbose,
                "desc": "                  aligning edge",
            },
            n_threads=self.n_threads,
        )

        # need to reorder the errors to match the order of the edges
        self.all_errors = np.array([x[1] for x in self._cache.values()])

        # Set error values above the threshold to infinity.
        for k, v in self._cache.items():
            if v[1] > self.max_error or np.any(np.abs(v[0]) > self.max_shift_pixels):
                self._cache[k] = (v[0], np.inf)

        self.cached_errors = self._cache.copy()  # save as a backup

    def _build_spanning_tree_gt(self):
        g = nxGraph()
        g.add_nodes_from(self.neighbors_graph)
        g.add_weighted_edges_from(
            (t1, t2, error) for (t1, t2), (_, error) in self.cached_errors.items() if np.isfinite(error)
        )

        # convert to a graph-tool graph
        gtG = nx2gt(g)

        spanning_tree = gtGraph(gtG)
        spanning_tree.clear_edges()

        # label the components in a property map
        c = label_components(gtG)[0]
        components = np.unique(c.a)

        centers = []
        for i in components:
            u = GraphView(gtG, vfilt=c.a == i)

            center = get_center_nodes(u)
            centers.append(center)
            vertices = list(u.vertices())
            for vertix in vertices:
                vlist, elist = shortest_path(u, center, vertix, weights=u.ep.weight)
                spanning_tree.add_edge_list(elist)

        remove_parallel_edges(spanning_tree)

        self.spanning_tree = spanning_tree
        self.centers_spanning_tree = centers

    def _build_spanning_tree_nxg(self):
        g = nx.Graph()
        g.add_nodes_from(self.neighbors_graph)
        g.add_weighted_edges_from((t1, t2, error) for (t1, t2), (_, error) in self._cache.items() if np.isfinite(error))
        spanning_tree = nx.Graph()
        spanning_tree.add_nodes_from(g)

        centers = []

        for c in nx.connected_components(g):
            cc = g.subgraph(c)
            center = nx.center(cc)[0]
            centers.append(center)
            paths = nx.single_source_dijkstra_path(cc, center).values()
            for path in paths:
                nx.add_path(spanning_tree, path)

        self.spanning_tree = spanning_tree
        self.centers_spanning_tree = centers

    def build_spanning_tree(self):
        if self.flavor == "graph-tool":
            print("using graph-tool to build spanning tree")
            self._build_spanning_tree_gt()
        if self.flavor == "networkx":
            print("using networkx to build spanning tree")
            self._build_spanning_tree_nxg()

    def _calculate_positions_gt(self):
        shifts = {}
        _components = []

        # label the components in a property map
        c = label_components(self.spanning_tree)[0]
        components = np.unique(c.a)

        for ix, i in enumerate(components):
            u = GraphView(self.spanning_tree, vfilt=c.a == i)
            nodes = list(u.get_vertices())
            _components.append(set(nodes))

            center = self.centers_spanning_tree[ix]

            shifts[center] = np.array([0, 0])

            if len(nodes) > 1:
                for edge in bfs_iterator(u, source=center):
                    source, dest = edge
                    source = int(source)
                    dest = int(dest)
                    if source not in shifts:
                        source, dest = dest, source
                    shift = self.register_pair(source, dest)[0]
                    shifts[dest] = shifts[source] + shift

        if shifts:
            self.shifts = np.array([s for _, s in sorted(shifts.items())])
            self.positions = self.metadata.positions + self.shifts
            self.components_spanning_tree = _components
        else:
            # TODO: fill in shifts and positions with 0x2 arrays
            raise NotImplementedError("No images")

    def _calculate_positions_nxg(self):
        shifts = {}
        for c in nx.connected_components(self.spanning_tree):
            cc = self.spanning_tree.subgraph(c)
            center = nx.center(cc)[0]
            shifts[center] = np.array([0, 0])
            for edge in nx.traversal.bfs_edges(cc, center):
                source, dest = edge
                if source not in shifts:
                    source, dest = dest, source
                shift = self.register_pair(source, dest)[0]
                shifts[dest] = shifts[source] + shift
        if shifts:
            self.shifts = np.array([s for _, s in sorted(shifts.items())])
            self.positions = self.metadata.positions + self.shifts
            self.components_spanning_tree = nx.connected_components(self.spanning_tree)
        else:
            # TODO: fill in shifts and positions with 0x2 arrays
            raise NotImplementedError("No images")

    def calculate_positions(self):
        if self.flavor == "graph-tool":
            self._calculate_positions_gt()
        if self.flavor == "networkx":
            self._calculate_positions_nxg()

    def fit_model(self):
        components = self.components_spanning_tree
        components = sorted(components, key=len, reverse=True)

        # Fit LR model on positions of largest connected component.
        cc0 = list(components[0])
        self.lr = sklearn.linear_model.LinearRegression()
        self.lr.fit(self.metadata.positions[cc0], self.positions[cc0])
        # Fix up degenerate transform matrix. This happens when the spanning
        # tree is completely edgeless or cc0's metadata positions fall in a
        # straight line. In this case we fall back to the identity transform.
        if np.linalg.det(self.lr.coef_) < 1e-3:
            # FIXME We should probably exit here, not just warn. We may provide
            # an option to force it anyway.
            warn_data("Could not align enough edges, proceeding anyway with original" " stage positions.")
            self.lr.coef_ = np.diag(np.ones(2))
            self.lr.intercept_ = np.zeros(2)

        # Adjust position of remaining components so their centroids match
        # the predictions of the model.
        for cc in components[1:]:
            nodes = list(cc)
            centroid_m = np.mean(self.metadata.positions[nodes], axis=0)
            centroid_f = np.mean(self.positions[nodes], axis=0)
            shift = self.lr.predict([centroid_m])[0] - centroid_f
            self.positions[nodes] += shift
        # Adjust positions and model intercept to put origin at 0,0.
        self.origin = self.positions.min(axis=0)
        self.positions -= self.origin
        self.lr.intercept_ -= self.origin
        self.centers = self.positions + self.metadata.size / 2


class ParallelMosaic(Mosaic):
    def __init__(
        self,
        aligner,
        shape,
        n_threads=20,
        channels=None,
        ffp_path=None,
        dfp_path=None,
        flip_mosaic_x=False,
        flip_mosaic_y=False,
        barrel_correction=None,
        verbose=False,
    ):
        super().__init__(
            aligner=aligner,
            shape=shape,
            channels=channels,
            ffp_path=ffp_path,
            dfp_path=dfp_path,
            flip_mosaic_x=flip_mosaic_x,
            flip_mosaic_y=flip_mosaic_y,
            barrel_correction=barrel_correction,
            verbose=verbose,
        )

        self.n_threads = n_threads

    def assemble_channel_parallel(
        self,
        channel,
        ch_index: int,
        out: np.ndarray | None = None,
        hdf5_path: str | None = None,
        tqdm_args=None,
        n_percent: int = 10,
    ):
        """This function assembles a single channel of the mosaic writing to the same HDF5 file being used as a mmap array in the backend."""
        if out is None:
            if hdf5_path is not None:
                out = mmap_array_from_path(hdf5_path)
            else:
                out = np.zeros(self.shape, self.dtype)
        else:
            if out.shape != self.shape:
                raise ValueError(f"out array shape {out.shape} does not match Mosaic shape {self.shape}")
            if hdf5_path is None:
                raise ValueError(
                    "if specifying an out array, you also need to pass the HDF5 path of the memory mapped temparray"
                )

        # Set up tqdm arguments if not provided
        if tqdm_args is None:
            tqdm_args = {
                "file": sys.stdout,
                "disable": not self.verbose,
                "desc": f"assembling channel {ch_index}",
                "total": len(self.aligner.positions),
            }

        total_positions = len(self.aligner.positions)
        update_interval = int(total_positions * (n_percent / 100))
        last_update = 0

        # Assemble channel with progress updates every n%
        with tqdm(**tqdm_args) as pbar:
            for si, position in enumerate(self.aligner.positions):
                img = self.aligner.reader.read(c=channel, series=si)
                img = self.correct_illumination(img, channel)
                utils.paste(out[ch_index, :, :], img, position, func=utils.pastefunc_blend)

                # Update progress bar every n%
                if si >= last_update + update_interval:
                    pbar.update(update_interval)
                    last_update = si

            # Ensure the progress bar reaches 100% at the end
            if pbar.n < total_positions:
                pbar.update(total_positions - pbar.n)

        # Memory-conserving axis flips
        if self.flip_mosaic_x:
            for i in range(len(out)):
                out[i] = out[i, ::-1]
        if self.flip_mosaic_y:
            for i in range(len(out) // 2):
                out[[i, -i - 1]] = out[[-i - 1, i]]

        return None
