from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
import numpy as np


def get_temporal_edge_times(dataset: PyGLinkPropPredDataset, src: int, dst: int, num_hops: int, mask=None):
    # Returns [time tensor 0-hop, time tensor 1-hop, ...]
    # Note: treats the graph as undirectional
    # Note: assumes no self-edges

    src_list = dataset.src.numpy()
    dst_list = dataset.dst.numpy()
    time_list = dataset.ts.numpy()

    if mask is not None:
        src_list = src_list[mask]
        dst_list = dst_list[mask]
        time_list = time_list[mask]

    # Computes bool map of all edges  u -> *  and  * -> u
    def get_conns(u):
        return np.logical_or(src_list == u, dst_list == u)

    # Computes bool map of all edges a->b or b->a
    def edge(a, b):
        v1 = np.logical_and(src_list == a, dst_list == b)
        v2 = np.logical_and(src_list == b, dst_list == a)
        return np.logical_or(v1, v2)

    # Given a bool map of edges, return all involved src/dest nodes
    def get_nodes(edge_map):
        return np.unique(np.concatenate((src_list[edge_map], dst_list[edge_map])))

    hop0 = edge(src, dst)
    frontier = get_nodes(hop0)  # just src, dst
    prev_hop = set(frontier.tolist())  # contains nodes which we are no longer interested in

    hops = [hop0]

    for i in range(1, num_hops + 1):
        hopi = np.zeros_like(hop0)
        for u in frontier:
            # find all v -> u  and  u -> v
            edge_map = get_conns(u)

            # Get all adjacent to u which are not in a previous hop and are not equal to u
            adj_list = [i for i in get_nodes(edge_map).tolist() if i!=u and i not in prev_hop]

            # Union hopi with the hops from u to next nodes
            for v in adj_list:
                hopi = np.logical_or(hopi, edge(u, v))

        # Ensure we do not visit anything in frontier again
        prev_hop.update(frontier.tolist())

        # Compute new frontier: edges in hopi which aren't yet explored
        # Note: get_nodes applies np.unique and prev_hop is a set, so assume_unique=True for speed
        frontier = np.setdiff1d(get_nodes(hopi), list(prev_hop), assume_unique=True)

        hops.append(hopi)

    times = [time_list[hop] for hop in hops]
    return times

