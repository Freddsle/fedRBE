"""
Interprets the performance reports of a dockerized fedRBE run.
Each client writes performance_report.json (batchcorrection/perf.py), the
spans of its work in the client role (client_compute, client_send,
client_await) and, for the coordinator, the aggregator role (agg_compute,
agg_gather, agg_broadcast). Spans of one role follow each other, spans of
different roles overlap: the coordinator's client_await of a round contains
its agg_gather, agg_compute and agg_broadcast of that round, as it awaits the
aggregated result while it creates it. Only the coordinator's agg_compute of
XtX_block without idx (the beta mask, before the first block is sent) lies
outside its client spans. As all containers run on the same host, the
timestamps are comparable between clients.

summarize_performance creates one row per client and span (except the
whole_protocol span around all others, its length is run_duration) with:
    duration: length of the span
    bytes: payload bytes the span sent, taken from the controller log
        (what the controller fetched from the container, for SMPC this is
        the payload before it is split into shards)
    delay: only for spans that end with received data:
        agg_gather of the coordinator: end of the gather minus the end of the
            last client_send of a client that sends via the network (the
            coordinator sends to itself without the network unless SMPC is used
            for the XtX blocks)
        client_await of the other clients: end of the await minus the end of
            the coordinator's agg_broadcast of the same round
    and per client: the run duration, peak and baseline resident memory of
    the app process and the container's cgroup peak memory, max anonymous
    memory and CPU time.
The coordinator's client_await contains its aggregation, so it has no delay,
its waiting for the other clients is agg_gather.
TODO: check and improve
"""
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd

# kind and round of the span around the whole protocol of a client
WHOLE_PROTOCOL = "whole_protocol"

def _sends_via_network(segment: dict, is_coordinator: bool, smpc: bool) -> bool:
    """Whether the controller fetches the data queued in this segment."""
    if segment["kind"] == "agg_broadcast":
        return True
    if segment["kind"] != "client_send":
        return False
    if not is_coordinator:
        return True
    # the coordinator sends to itself without the controller unless SMPC is
    # used, which fedrbe.py only does for the XtX blocks
    return smpc and segment["round"] == "XtX_block"


def _key(segment: dict) -> Tuple[str, Optional[int]]:
    return segment["round"], segment["idx"]


def summarize_performance(client_names: List[str], reports: List[dict],
                          container_stats: Optional[List[dict]] = None,
                          controller_traffic: Optional[List[List[dict]]] = None
                          ) -> pd.DataFrame:
    """
    Args:
        client_names: Client names in the order of the reports
        reports: The parsed performance_report.json per client
        container_stats: Per client (same order), the container stats of
            featurecloud_api_extension.ExperimentResult.container_stats
        controller_traffic: Per client (same order), the controller's fetches
            [{"time": ..., "bytes": ...}, ...] in the order they happened, from
            featurecloud_api_extension.ExperimentResult.controller_traffic
    Returns:
        DataFrame with one row per client and span, see the module docstring
    Raises:
        ValueError: if spans of the same role overlap, a client has not exactly
            one whole_protocol span containing all its other spans, there is
            not exactly one coordinator or the number of fetches does not
            match the sends
    """
    if len(client_names) != len(reports):
        raise ValueError("Need one performance report per client")
    coordinators = [idx for idx, report in enumerate(reports) if report["is_coordinator"]]
    if len(coordinators) != 1:
        raise ValueError(f"Expected exactly one coordinator, got {len(coordinators)}")
    coordinator = reports[coordinators[0]]
    smpc = bool(coordinator["info"]["smpc"])

    for name, report in zip(client_names, reports):
        for role in ("client_", "agg_"):
            spans = sorted((seg for seg in report["segments"] if seg["kind"].startswith(role)),
                           key=lambda seg: seg["t_start"])
            for previous, current in zip(spans, spans[1:]):
                if current["t_start"] < previous["t_end"]:
                    raise ValueError(f"Spans of {name} overlap: {previous} and {current}")

    # end of the coordinator's broadcast per round, used for the downlink delay
    broadcast_end: Dict[tuple, float] = {
        _key(seg): seg["t_end"] for seg in coordinator["segments"]
        if seg["kind"] == "agg_broadcast"}
    # end of the last network send per round, used for the uplink delay
    last_send_end: Dict[tuple, float] = dict()
    for report in reports:
        for seg in report["segments"]:
            if seg["kind"] == "client_send" and \
                    _sends_via_network(seg, report["is_coordinator"], smpc):
                last_send_end[_key(seg)] = max(last_send_end.get(_key(seg), seg["t_end"]),
                                               seg["t_end"])

    rows = list()
    for idx, (name, report) in enumerate(zip(client_names, reports)):
        is_coordinator = report["is_coordinator"]
        whole = [seg for seg in report["segments"] if seg["kind"] == WHOLE_PROTOCOL]
        if len(whole) != 1:
            raise ValueError(f"Client {name}: expected one {WHOLE_PROTOCOL} span, got {len(whole)}")
        whole = whole[0]
        # spans are stored in the order they ended
        segments = sorted((seg for seg in report["segments"] if seg["kind"] != WHOLE_PROTOCOL),
                          key=lambda seg: seg["t_start"])
        outside = [seg for seg in segments
                   if seg["t_start"] < whole["t_start"] or seg["t_end"] > whole["t_end"]]
        if outside:
            raise ValueError(f"Client {name}: spans outside of {WHOLE_PROTOCOL}: {outside}")
        network_sends = [seg for seg in segments
                         if _sends_via_network(seg, is_coordinator, smpc)]
        fetched = None
        if controller_traffic is not None:
            fetched = controller_traffic[idx]
            if len(fetched) != len(network_sends):
                raise ValueError(
                    f"Client {name}: the controller fetched {len(fetched)} payloads, "
                    f"but the timeline has {len(network_sends)} network sends")
        # the controller fetches in the order the data was queued
        sent_bytes = {id(seg): fetch["bytes"] for seg, fetch
                      in zip(network_sends, fetched or [])}

        stats = container_stats[idx] if container_stats is not None else dict()
        per_client = {
            "client": name,
            "coordinator": is_coordinator,
            "smpc": smpc,
            "run_duration": whole["t_end"] - whole["t_start"],
            "app_pre_fedrbe_peak_rss_bytes": report["pre_fedrbe_peak_rss_kib"] * 1024,
            "app_baseline_rss_bytes": report["baseline_rss_kib"] * 1024,
            "app_peak_rss_bytes": report["peak_rss_kib"] * 1024,
            "container_mem_peak_bytes": stats.get("memory_peak"),
            "container_max_anon_bytes": stats.get("max_anon"),
            "container_cpu_usec": stats.get("cpu_usec"),
        }
        for seg in segments:
            delay = None
            if seg["kind"] == "agg_gather" and _key(seg) in last_send_end:
                delay = seg["t_end"] - last_send_end[_key(seg)]
            elif seg["kind"] == "client_await" and not is_coordinator:
                delay = seg["t_end"] - broadcast_end[_key(seg)]
            rows.append({
                **per_client,
                "role": "aggregator" if seg["kind"].startswith("agg_") else "client",
                "kind": seg["kind"],
                "round": seg["round"],
                "idx": seg["idx"],
                "t_start": seg["t_start"],
                "t_end": seg["t_end"],
                "duration": seg["t_end"] - seg["t_start"],
                "rss_end_bytes": seg["rss_end_kib"] * 1024,
                "bytes": sent_bytes.get(id(seg)),
                "delay": delay,
            })
    return pd.DataFrame(rows)


def load_reports(client_files: List[dict], filename: str) -> Optional[List[dict]]:
    """
    Parses the performance report of every client, None if any client has no
    report (e.g. an app image built before the reports were added).
    """
    if any(filename not in outputs for outputs in client_files):
        return None
    return [json.loads(outputs[filename]) for outputs in client_files]
