try:
    from graph_tool import Graph
    from graph_tool.topology import shortest_distance
except ImportError:
    Graph = None
    shortest_distance = None

from networkx import Graph as nxGraph


# calcultion of centers is equivalent to method in networkx: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.distance_measures.center.html
def get_center_nodes(g):
    """Calculate the center node of the graph.

    The center is the set of nodes with eccentricity equal to radius.

    Parameters
    ----------
    g : graph-tool Graph
        The graph for which to calculate the center node.

    Returns
    -------
    centers : int
        index of the center node of the graph.
    """
    if shortest_distance is None:
        raise ImportError("graph-tool is required for this function")

    nodes = g.get_vertices()

    if len(nodes) > 2:
        distances = shortest_distance(g)
        eccentricities = [x.a.max() for x in distances]

        centers = nodes[eccentricities.index(min(eccentricities))]

        return centers

    # if only one node or two nodes are present in the graph the one with the lower index is returned
    else:
        return nodes[0]


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.

    Parameters
    ----------
    nxG : networkx.Graph
        The networkx graph to convert.

    Returns
    -------
    gtG : graph-tool.Graph
        The graph-tool graph.
    """
    node_list = list(nxG.nodes())
    edge_list = list(nxG.edges(data=True))
    edge_list = [(x, y, weight["weight"]) for x, y, weight in edge_list]
    gtG = Graph(edge_list, eprops=[("weight", "float")], directed=nxG.is_directed())

    vertices = gtG.get_vertices()
    for _missing_node in [x for x in node_list if x not in vertices]:
        gtG.add_vertex()

    return gtG


def gt2nx(gtG):
    """
    Converts a graph-tool graph to a NetworkX graph.

    Parameters
    ----------
    gtG : graph-tool.Graph
        The graph-tool graph to convert.

    Returns
    -------
    nxG : networkx.Graph
        The NetworkX graph.
    """
    # Create a new NetworkX graph
    nxG = nxGraph()
    edges = gtG.get_edges()
    nxG.add_edges_from(edges)
    return nxG
