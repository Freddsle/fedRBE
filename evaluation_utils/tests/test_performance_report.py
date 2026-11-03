"""Checks the fedRBE timeline recorder and the interpretation of its reports."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import yaml

from batchcorrection.fedrbe import run_fedrbe
from batchcorrection.perf import PerfRecorder, REPORT_FILENAME
from evaluation_utils import featurecloud_api_extension as api
from evaluation_utils.performance_report import summarize_clients, summarize_performance

sys.path.append(str(Path(__file__).resolve().parents[2] / "evaluation_classification_after_correction"))
from fc_fed_forest_simple_app.fc_fedlearnsim_helper.run_app_simulation import (  # noqa: E402
    run_simulation_native,
)

NAMES = ["client_a", "client_b", "client_c"]
N_BLOCKS = 2
# seconds, maximal gap between two client spans that follow each other. The
# clients of the native simulation are threads sharing the GIL, so the next
# start can be delayed by the other clients.
GAP = 0.05
RSS_TOLERANCE_KIB = 1024
# kind, round, idx of the client spans of every client, in order
CLIENT_SPANS = [
    ("client_compute", "commonVariables", None), ("client_send", "commonVariables", None),
    ("client_await", "commonVariables", None),
    ("client_compute", "commonGenes", None), ("client_send", "commonGenes", None),
    ("client_await", "commonGenes", None),
    ("client_compute", "XtX_block", None),
    ("client_send", "XtX_block", 0), ("client_await", "XtX_block", 0),
    ("client_send", "XtX_block", 1), ("client_await", "XtX_block", 1),
    ("client_await", "beta", None), ("client_compute", "final", None),
]


def write_clients(root: Path, smpc: bool):
    """Three clients with three features, corrected in two feature blocks."""
    covariate = np.array([0, 1] * 4, dtype=float)
    for number, name in enumerate(NAMES):
        directory = root / "input" / name
        directory.mkdir(parents=True)
        values = np.array([10 + 3 * number + 2 * covariate + np.arange(8) / 10,
                           20 - 2 * number + covariate / 2 + np.arange(8) ** 2 / 20,
                           5 + number + np.arange(8) / 3])
        data = pd.DataFrame(values, index=["feature_a", "feature_b", "feature_c"],
                            columns=[f"{name}_sample_{index}" for index in range(8)])
        data.loc["condition"] = covariate
        data.to_csv(directory / "data.tsv", sep="\t")
        config = {"data_filename": "data.tsv", "expression_file_flag": True, "index_col": 0,
                  "covariates": ["condition"], "separator": "\t", "smpc": smpc,
                  "feature_block_size": 2}
        (directory / "config.yml").write_text(
            yaml.safe_dump({"flimmaBatchCorrection": config}), encoding="utf-8")
        (root / "output" / name).mkdir(parents=True)


def run_with_reports(root: Path, smpc: bool):
    """Runs the native simulation, every client writes its performance report."""
    write_clients(root, smpc)

    def run_client(protocol, inputfolder, outputfolder):
        perf = PerfRecorder()
        run_fedrbe(protocol, inputfolder, outputfolder, perf=perf)
        perf.write(outputfolder, client_id=protocol.id,
                   is_coordinator=protocol.is_coordinator, clients=protocol.clients)

    with contextlib.redirect_stdout(io.StringIO()):
        run_simulation_native([str(root / "input" / name) for name in NAMES],
                              [str(root / "output" / name) for name in NAMES],
                              None, run_client, client_ids=NAMES)
    return [json.loads((root / "output" / name / REPORT_FILENAME).read_text())
            for name in NAMES]


def network_sends(report):
    """Number of payloads the controller would fetch from this client."""
    if not report["is_coordinator"]:
        return 2 + N_BLOCKS  # commonVariables, commonGenes, XtX blocks
    # broadcasts of commonVariables, commonGenes, every block ack and beta,
    # plus the SMPC XtX blocks
    return 3 + N_BLOCKS + (N_BLOCKS if report["info"]["smpc"] else 0)


class PerformanceReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_timeline_separates_client_and_aggregator_roles(self):
        reports = run_with_reports(self.root, smpc=False)
        for report in reports:
            segments = sorted(report["segments"], key=lambda seg: seg["t_start"])
            client = [seg for seg in segments if seg["kind"].startswith("client_")]
            self.assertEqual([(seg["kind"], seg["round"], seg["idx"]) for seg in client],
                             CLIENT_SPANS)
            # VmRSS of a multithreaded process is summed from per CPU
            # counters and can be off by a few pages
            self.assertGreaterEqual(report["peak_rss_kib"] + RSS_TOLERANCE_KIB,
                                    report["final_rss_kib"])
            aggregator = [seg for seg in segments if seg["kind"].startswith("agg_")]
            for previous, current in zip(client, client[1:]):
                gap = current["t_start"] - previous["t_end"]
                self.assertGreaterEqual(gap, 0)
                # only the coordinator's beta mask lies between two client spans
                if not (report["is_coordinator"] and current["kind"] == "client_send"
                        and current["idx"] == 0):
                    self.assertLess(gap, GAP)
            awaits = [(seg["round"], seg["idx"]) for seg in segments
                      if seg["kind"] == "client_await"]
            self.assertEqual(awaits, [("commonVariables", None), ("commonGenes", None),
                                      ("XtX_block", 0), ("XtX_block", 1), ("beta", None)])
            if not report["is_coordinator"]:
                self.assertEqual(aggregator, [])
                continue
            gathers = [(seg["round"], seg["idx"]) for seg in aggregator
                       if seg["kind"] == "agg_gather"]
            self.assertEqual(gathers, [("commonVariables", None), ("commonGenes", None),
                                       ("XtX_block", 0), ("XtX_block", 1)])
            # the coordinator aggregates while it awaits the aggregated result
            await_spans = {(seg["round"], seg["idx"]): seg for seg in client
                           if seg["kind"] == "client_await"}
            for seg in aggregator:
                key = (seg["round"], seg["idx"])
                if key == ("XtX_block", None):
                    continue  # the beta mask, before the first block is sent
                self.assertLessEqual(await_spans[key]["t_start"], seg["t_start"])
                self.assertLessEqual(seg["t_end"], await_spans[key]["t_end"])
            # the beta mask is aggregator work before the first block
            self.assertIn(("agg_compute", "XtX_block", None),
                          [(seg["kind"], seg["round"], seg["idx"]) for seg in aggregator])

    def test_summary_maps_fetched_payloads_and_delays(self):
        for smpc in (False, True):
            with self.subTest(smpc=smpc):
                root = self.root / str(smpc)
                reports = run_with_reports(root, smpc=smpc)
                traffic = [[{"time": 0, "bytes": 100 * (idx + 1) + send}
                            for send in range(network_sends(report))]
                           for idx, report in enumerate(reports)]
                stats = [{"memory_peak": 3, "max_anon": 2, "cpu_usec": 1}] * len(NAMES)
                summary = summarize_performance(NAMES, reports, stats, traffic)

                for name, rows in summary.groupby("client"):
                    # the client spans plus the beta mask cover the whole run
                    covered = rows[(rows["role"] == "client")
                                   | (rows["idx"].isna() & (rows["round"] == "XtX_block"))]
                    self.assertAlmostEqual(covered["duration"].sum(),
                                           rows["run_duration"].iloc[0], delta=GAP)
                    sent = rows.dropna(subset=["bytes"])
                    self.assertEqual(len(sent), len(traffic[NAMES.index(name)]))
                coordinator = summary[summary["coordinator"]]
                self.assertTrue(coordinator["kind"].isin(["agg_broadcast", "client_send"])
                                .loc[coordinator["bytes"].notna()].all())
                gathers = coordinator[coordinator["kind"] == "agg_gather"]
                self.assertTrue(gathers["delay"].notna().all(), gathers["delay"])
                awaits = summary[(summary["kind"] == "client_await") & ~summary["coordinator"]]
                # a client thread of the simulation can receive the broadcast
                # before the coordinator thread ends its agg_broadcast span
                self.assertTrue((awaits["delay"] > -GAP).all(), awaits["delay"].min())
                self.assertEqual(len(awaits), (len(NAMES) - 1) * (3 + N_BLOCKS))

                self.assertFalse((summary["kind"] == "whole_protocol").any())
                for name, report in zip(NAMES, reports):
                    whole = next(seg for seg in report["segments"]
                                 if seg["kind"] == "whole_protocol")
                    self.assertEqual(
                        summary.loc[summary["client"] == name, "run_duration"].iloc[0],
                        whole["t_end"] - whole["t_start"])

                traffic[1] = traffic[1][1:]
                with self.assertRaisesRegex(ValueError, "fetched"):
                    summarize_performance(NAMES, reports, stats, traffic)
                reports[0]["segments"] = [seg for seg in reports[0]["segments"]
                                          if seg["kind"] != "whole_protocol"]
                with self.assertRaisesRegex(ValueError, "whole_protocol"):
                    summarize_performance(NAMES, reports, stats)

    def test_client_summary_splits_runtime_into_compute_aggregation_and_waiting(self):
        reports = run_with_reports(self.root, smpc=False)
        stats = [{"memory_peak": 3, "max_anon": 2, "cpu_usec": 1}] * len(NAMES)
        spans = summarize_performance(NAMES, reports, stats)
        clients = summarize_clients(spans).set_index("client")

        self.assertEqual(list(clients.index), NAMES)
        self.assertTrue((clients["n_samples"] == 8).all())
        self.assertTrue((clients["n_features"] == 3).all())
        self.assertTrue((clients["n_features_used"] == 3).all())
        total = clients["client_compute"] + clients["aggregation"] + clients["waiting"]
        np.testing.assert_allclose(total, clients["run_duration"])
        self.assertTrue((clients["waiting"] >= 0).all())
        self.assertTrue((clients["client_compute"] > 0).all())
        coordinator = clients["coordinator"]
        self.assertTrue((clients.loc[~coordinator, "aggregation"] == 0).all())
        self.assertTrue((clients.loc[coordinator, "aggregation"] > 0).all())
        for name, report in zip(NAMES, reports):
            sends = sum(seg["t_end"] - seg["t_start"] for seg in report["segments"]
                        if seg["kind"] in ("client_compute", "client_send"))
            self.assertAlmostEqual(clients.loc[name, "client_compute"], sends)

    def test_controller_traffic_reads_fetches_of_the_test_window(self):
        logs = self.root / "logs"
        logs.mkdir()
        lines = [
            {"component": "LOCAL", "msg": "http://172.18.0.5:9000/data: Fetched 7 bytes",
             "time": "2026-10-27T10:24:08Z"},
            {"component": "LOCAL", "msg": "http://172.18.0.6:9000/data: Fetched 9 bytes",
             "time": "2026-10-27T10:24:09Z"},
            {"component": "LOCAL", "msg": "received 9 bytes from Client 77ab",
             "time": "2026-10-27T10:24:09Z"},
            {"component": "LOCAL", "msg": "http://172.18.0.5:9000/data: Fetched 11 bytes",
             "time": "2026-10-27T10:30:00Z"},
        ]
        (logs / "log-2026-10-27_10-00-00.log").write_text(
            "\n".join(json.dumps(line) for line in lines) + "\nnot json\n")
        window_start = pd.Timestamp("2026-10-27T10:24:07Z").timestamp()
        sampler = SimpleNamespace(
            t_start=window_start, t_end=window_start + 10,
            ip_to_name=lambda: {"172.18.0.5": "fc_a", "172.18.0.6": "fc_b"})
        experiment = api.Experiment(name="fixture", clients=[], app_image_name="fixture",
                                    fc_data_dir=str(self.root))
        traffic = experiment._controller_traffic(sampler, ["fc_b", "fc_a"])
        self.assertEqual([[fetch["bytes"] for fetch in client] for client in traffic],
                         [[9], [7]])


if __name__ == "__main__":
    unittest.main()
