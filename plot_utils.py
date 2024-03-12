from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
import numpy as np
from numpy import ndarray
from typing import Tuple


def calculate_average_step_difference_full_range(
    timestamps_by_hops: list[ndarray],
    preds: ndarray,
    pred_timestamps: ndarray,
    hop_threshold: int = 1,
) -> float:
    aggregated_timestamps = np.concatenate(
        timestamps_by_hops[: hop_threshold + 1], axis=0
    )
    print("Full range: ")
    # print(f"Shape of aggregated_timestamps: {aggregated_timestamps.shape}")
    aggregated_timestamps = np.sort(aggregated_timestamps, axis=0)

    time_to_preds = {t: v for t, v in zip(pred_timestamps, preds)}

    aggregated_step_diff = np.sum(np.abs(preds[1:] - preds[:-1]))
    total_step_cnt = len(preds) - 1

    for t in aggregated_timestamps:
        assert t in time_to_preds
        total_step_cnt -= 1
        aggregated_step_diff -= np.abs(time_to_preds[t + 1] - time_to_preds[t])

    print(
        f"Total steps between first and last pred: {pred_timestamps[-1] - pred_timestamps[0]}"
    )
    print(f"Total step count: {total_step_cnt}")
    return aggregated_step_diff / total_step_cnt


def calculate_average_step_difference(
    timestamps_by_hops: list[ndarray],
    preds: ndarray,
    pred_timestamps: ndarray,
    hop_threshold: int = 1,
) -> float:
    aggregated_timestamps = np.concatenate(
        timestamps_by_hops[: hop_threshold + 1], axis=0
    )
    print(f"Shape of aggregated_timestamps: {aggregated_timestamps.shape}")
    aggregated_timestamps = np.sort(aggregated_timestamps, axis=0)

    event_index = 0
    lower_bound = aggregated_timestamps[event_index]
    upper_bound = aggregated_timestamps[event_index + 1]

    print(f"Number of predictions: {len(preds)}")
    pred_index = 0

    accumulated_step_difference = 0

    while pred_timestamps[pred_index] < lower_bound:
        pred_index += 1
    assert pred_timestamps[pred_index] == lower_bound

    skipped_step_cnt = 0
    processed_step_cnt = 0

    while event_index < len(aggregated_timestamps) - 1:
        lower_bound = aggregated_timestamps[event_index]
        upper_bound = aggregated_timestamps[event_index + 1]
        # start from the second after the lower bound event
        if pred_timestamps[pred_index] == lower_bound:
            # skip because this update is directly caused by the event
            pred_index += 1
            skipped_step_cnt += 1

        while pred_timestamps[pred_index] < upper_bound:
            if pred_index + 1 == len(preds):
                break
            accumulated_step_difference += np.abs(
                preds[pred_index + 1] - preds[pred_index]
            )
            pred_index += 1
            processed_step_cnt += 1

        event_index += 1

    print(
        f"Total steps between first and last event: {aggregated_timestamps[-1] - aggregated_timestamps[0]}"
    )
    print(f"Skipped step count: {skipped_step_cnt}")
    print(f"Processed step count: {processed_step_cnt}")
    return accumulated_step_difference / processed_step_cnt


def total_variation_per_unit_time(
    timestamps_by_hops: list[ndarray], preds: ndarray, pred_timestamps: ndarray
) -> Tuple[float, float]:
    # Ensure sorted chronologically
    sort_inds = np.argsort(pred_timestamps)
    preds = preds[sort_inds]
    pred_timestamps = pred_timestamps[sort_inds]

    max_time = pred_timestamps[-1]

    # Calculate total variation over the whole period
    diffs = np.sum(np.abs(preds[1:] - preds[:-1]))
    time_length = float(pred_timestamps[-1] - pred_timestamps[0])

    # Discount variation due to events at different hop levels
    for ts in timestamps_by_hops:
        # inds = the indices of the measurements taken before (or at the same time as) the event
        #  inds+1 = the first indices where the change has taken effect
        # We therefore discount the (inds -> inds+1) edges
        inds = np.searchsorted(pred_timestamps, ts, side="right") - 1
        inds = np.unique(inds)  # only discount any step at most once
        inds = inds[inds < preds.shape[0] - 1]  # there is no step following the final timestep
        diffs -= np.sum(
            np.abs(preds[inds + 1] - preds[inds])
        )  # discount all inds->inds+1 probability jumps
        time_length -= np.sum(
            pred_timestamps[inds + 1] - pred_timestamps[inds]
        )  # and also remove these time periods

    return diffs.item(), (diffs / time_length).item()


def get_temporal_edge_times(
    dataset: PyGLinkPropPredDataset, src: int, dst: int, num_hops: int, mask=None
):
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
    prev_hop = set(
        frontier.tolist()
    )  # contains nodes which we are no longer interested in

    hops = [hop0]

    for i in range(1, num_hops + 1):
        hopi = np.zeros_like(hop0)
        for u in frontier:
            # find all v -> u  and  u -> v
            edge_map = get_conns(u)

            # Get all adjacent to u which are not in a previous hop and are not equal to u
            adj_list = [
                i for i in get_nodes(edge_map).tolist() if i != u and i not in prev_hop
            ]

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
