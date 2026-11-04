"""Bounded protocol checks for SMPC agreement, invalid runs and coefficient broadcasts."""

import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from batchcorrection import classes, fedrbe, perf
from batchcorrection.classes import client, coordinator_utils


def load_states():
    engine = importlib.import_module("FeatureCloud.app.engine.app")
    path = Path(__file__).resolve().parents[1] / "states.py"
    spec = importlib.util.spec_from_file_location("fedrbe_states_under_test", path)
    states = importlib.util.module_from_spec(spec)
    # Keep the app's container imports and state registration local to this test.
    aliases = {
        "classes": classes,
        "classes.client": client,
        "classes.coordinator_utils": coordinator_utils,
        "fedrbe": fedrbe,
        "perf": perf,
    }
    with patch.dict(sys.modules, aliases), patch.object(engine, "app", engine.App()):
        spec.loader.exec_module(states)
    return states, engine


class StopProtocol(Exception):
    """Raised by a mock to end the protocol after the checked step."""


class SmpcAgreementTests(unittest.TestCase):
    def make_protocol(self, local_smpc, remote_smpc):
        protocol = SimpleNamespace(
            id="site_a",
            is_coordinator=True,
            clients=["site_a", "site_b"],
            send_data_to_coordinator=Mock(),
            broadcast_data=Mock(),
            await_data=Mock(return_value=(["covariate"], 2)),
            gather_data=Mock(return_value=[
                (["site_a"], ["covariate"], local_smpc, None),
                (["site_b"], ["covariate"], remote_smpc, None),
            ]),
        )
        cohort = SimpleNamespace(
            smpc=local_smpc,
            feature_block_size=None,
            separator="\t",
            hash2feature={"feature": "feature_name"},
            hash2variable={"covariate": "covariate_name"},
            batch_labels=["site_a"],
            min_samples=0,
            n_samples=4,
            feature_names=["feature"],
            position=0,
            reference_batch=False,
            config_based_init=Mock(),
            exclude_high_leverage=Mock(return_value=0),
            get_batch_feature_presence_info=Mock(
                return_value={"site_a": ["feature"]}
            ),
        )
        return protocol, cohort

    def test_mixed_settings_stop_before_broadcast_or_feature_exchange(self):
        for local_smpc in (False, True):
            with self.subTest(coordinator_smpc=local_smpc):
                protocol, cohort = self.make_protocol(local_smpc, not local_smpc)
                with patch.object(fedrbe, "Client", return_value=cohort):
                    with self.assertRaisesRegex(RuntimeError, "SMPC|smpc"):
                        fedrbe.run_fedrbe(protocol, "input", "output")
                protocol.broadcast_data.assert_not_called()

    def test_mismatched_block_sizes_stop_before_broadcast_or_feature_exchange(self):
        for local_size, remote_size in ((None, 2), (2, None), (2, 3)):
            with self.subTest(local_size=local_size, remote_size=remote_size):
                protocol, cohort = self.make_protocol(True, True)
                cohort.feature_block_size = local_size
                protocol.gather_data.return_value = [
                    (["site_a"], ["covariate"], True, local_size),
                    (["site_b"], ["covariate"], True, remote_size),
                ]
                with patch.object(fedrbe, "Client", return_value=cohort):
                    with self.assertRaisesRegex(RuntimeError, "feature_block_size"):
                        fedrbe.run_fedrbe(protocol, "input", "output")
                protocol.broadcast_data.assert_not_called()
                cohort.get_batch_feature_presence_info.assert_not_called()

    def test_matching_boolean_settings_continue_initialization(self):
        for smpc in (False, True):
            with self.subTest(smpc=smpc):
                protocol, cohort = self.make_protocol(smpc, smpc)
                with patch.object(fedrbe, "Client", return_value=cohort), \
                        patch.object(fedrbe, "select_common_features_variables",
                                     side_effect=StopProtocol):
                    with self.assertRaises(StopProtocol):
                        fedrbe.run_fedrbe(protocol, "input", "output")
                self.assertIs(
                    protocol.send_data_to_coordinator.call_args_list[0].args[0][2], smpc,
                )
                self.assertIsNone(protocol.send_data_to_coordinator.call_args_list[0].args[0][3])
                protocol.broadcast_data.assert_called_once()

    def test_solver_error_bound_follows_smpc_settings(self):
        for smpc, exponent, expected in ((False, 12, 0.0), (True, 13, 2e-13)):
            with self.subTest(smpc=smpc, exponent=exponent):
                with patch.object(fedrbe, "SMPC_EXPONENT", exponent):
                    self.assertEqual(fedrbe.smpc_error_bound(smpc, 2), expected)

    def test_empty_batch_stops_the_participant_with_the_threshold_rule(self):
        protocol, cohort = self.make_protocol(True, True)
        protocol.is_coordinator = False
        cohort.get_batch_feature_presence_info.return_value = {
            "site_a|first": [], "site_a|second": ["feature"],
        }
        with patch.object(fedrbe, "Client", return_value=cohort):
            with self.assertRaisesRegex(
                RuntimeError,
                r"batches \['site_a\|first'\]\..*max\(min_samples=0, 2 batches \+ 1 covariates \+ 1\)",
            ):
                fedrbe.run_fedrbe(protocol, "input", "output")
        # Only the label exchange happened; presence information was never sent.
        self.assertEqual(protocol.send_data_to_coordinator.call_count, 1)
        protocol.broadcast_data.assert_not_called()

    def test_large_statistics_stop_before_the_smpc_send(self):
        # At exponent 12 the two-client limit is about 4.61e6, for both XtX and XtY.
        for smpc, value, stops in ((True, 5e6, True), (True, 4e6, False), (False, 5e6, False)):
            for statistic in ("XtX", "XtY"):
                with self.subTest(smpc=smpc, value=value, statistic=statistic):
                    protocol, cohort = self.make_protocol(smpc, smpc)
                    protocol.is_coordinator = False
                    protocol.await_data.side_effect = [
                        (["covariate"], 2), (["feature"], protocol.clients),
                    ]
                    cohort.validate_inputs = Mock()
                    cohort.set_data = Mock()
                    cohort.create_design = Mock(return_value=None)
                    cohort.feature_names = ["feature"]
                    cohort.data = MagicMock()
                    cohort.design = pd.DataFrame(np.ones((4, 3)))
                    cohort._check_consistency_designfile = Mock()
                    xtx = np.full((1, 3, 3), value if statistic == "XtX" else 1.0)
                    xty = np.full((1, 3), value if statistic == "XtY" else 1.0)
                    cohort.compute_XtX_XtY = Mock(return_value=(xtx, xty, ""))
                    # End the successful path at its statistics send.
                    protocol.send_data_to_coordinator.side_effect = [None, None, StopProtocol]
                    with patch.object(fedrbe, "Client", return_value=cohort), \
                            contextlib.redirect_stdout(io.StringIO()):
                        if stops:
                            with self.assertRaisesRegex(
                                RuntimeError, r"SMPC limit of 4\.61e\+06 for 2 clients",
                            ):
                                fedrbe.run_fedrbe(protocol, "input", "output")
                            self.assertEqual(protocol.send_data_to_coordinator.call_count, 2)
                        else:
                            with self.assertRaises(StopProtocol):
                                fedrbe.run_fedrbe(protocol, "input", "output")
                            self.assertEqual(protocol.send_data_to_coordinator.call_count, 3)
                            self.assertIs(
                                protocol.send_data_to_coordinator.call_args.kwargs["use_smpc"], smpc,
                            )

    def test_initial_state_runs_the_protocol_with_the_app_exponent(self):
        states, engine = load_states()
        state = states.InitialState()
        state._app = engine.App()
        with patch.object(states, "run_fedrbe") as run, \
                patch.object(states, "PerfRecorder") as recorder:
            self.assertEqual(state.run(), "terminal")
        run.assert_called_once_with(state, states.INPUT_FOLDER, states.OUTPUT_FOLDER,
                                    perf=recorder.return_value)
        recorder.return_value.write.assert_called_once()
        self.assertEqual(state._app.default_smpc["exponent"], fedrbe.SMPC_EXPONENT)

    def test_statistics_follow_smpc_and_only_batch_coefficients_are_broadcast(self):
        cohorts = ["site_a", "site_b"]
        for smpc in (False, True):
            with self.subTest(smpc=smpc):
                protocol = SimpleNamespace(
                    id="site_a",
                    is_coordinator=True,
                    clients=cohorts,
                    send_data_to_coordinator=Mock(),
                    broadcast_data=Mock(),
                    gather_data=Mock(side_effect=[
                        [(["site_a"], ["covariate"], smpc, None),
                         (["site_b"], ["covariate"], smpc, None)],
                        [],  # feature information, consumed by the patched feature selection
                        [],  # XtX and XtY, consumed by the patched solver
                    ]),
                    await_data=Mock(side_effect=[
                        (["covariate"], 2), (["feature"], cohorts), None, np.array([[3.]]),
                    ]),
                )
                cohort = MagicMock(
                    smpc=smpc, feature_block_size=None, separator="\t", report="",
                    min_samples=0, position=0,
                    reference_batch=False, batch_labels=["site_a"], feature_names=["feature"],
                    hash2feature={"feature": "feature_name"}, hash2variable={"covariate": "covariate_name"},
                    # Design columns: intercept, one covariate, one batch column (two batches).
                    design=pd.DataFrame(np.ones((4, 3))),
                )
                cohort.create_design.return_value = None
                cohort.compute_XtX_XtY.return_value = (np.ones((1, 3, 3)), np.ones((1, 3)), "")
                with patch.object(fedrbe, "Client", return_value=cohort), \
                        patch.object(fedrbe, "select_common_features_variables",
                                     return_value=(["feature"], np.ones((1, 2)), cohorts)), \
                        patch.object(fedrbe, "compute_beta",
                                     return_value=np.array([[1., 2., 3.]])) as solve, \
                        tempfile.TemporaryDirectory() as output:
                    fedrbe.run_fedrbe(protocol, "input", output)
                # XtX/XtY travel with the configured SMPC setting and the solver gets the matching bound
                self.assertIs(protocol.send_data_to_coordinator.call_args.kwargs["use_smpc"], smpc)
                self.assertIs(protocol.gather_data.call_args.kwargs["use_smpc"], smpc)
                self.assertEqual(solve.call_args.kwargs["xtx_error"],
                                 fedrbe.smpc_error_bound(smpc, len(cohorts)))
                self.assertEqual(solve.call_args.kwargs["n_batch_columns"], 1)
                beta_broadcast = protocol.broadcast_data.call_args
                self.assertEqual(beta_broadcast.kwargs["memo"], "beta")
                np.testing.assert_array_equal(beta_broadcast.args[0], [[3.]])

    def test_feature_blocks_preserve_results_and_wait(self):
        features = ["f4", "f0", "f3", "f1", "f2"]
        presence = np.array([[1, 1], [1, 1], [0, 1], [1, 1], [1, 0]])
        mask = coordinator_utils.create_beta_mask(presence, 5, 3)
        xtx = np.tile(np.diag([2., 3., 4.]), (5, 1, 1))
        xtx[mask[:, -1], -1, -1] = 0
        xty = np.einsum("fij,fj->fi", xtx, np.arange(15.).reshape(5, 3) + 1)
        reference = coordinator_utils.compute_beta(
            [[xtx, xty]], 5, 3, mask, n_batch_columns=1,
        )[:, -1:]
        cases = [(None, True, False), (1, True, False), (2, True, False),
                 (10, True, False), (2, False, False), (2, True, True)]
        cases = [(smpc, *case) for smpc in (False, True) for case in cases]
        for smpc, size, coordinator, fail_later in cases:
            with self.subTest(smpc=smpc, size=size, coordinator=coordinator,
                              fail_later=fail_later):
                width = size or len(features)
                blocks = [[xtx[i:i + width].copy(), xty[i:i + width]]
                          for i in range(0, len(features), width)]
                if fail_later:
                    blocks[1][0][0, 0, 0] = 0  # Second-block rank failure.
                protocol, _ = self.make_protocol(smpc, smpc)
                protocol.is_coordinator = coordinator
                handshake = [(*item[:3], size) for item in protocol.gather_data.return_value]
                protocol.gather_data.side_effect = [handshake, [], *[[block] for block in blocks]]
                cohort = MagicMock(
                    smpc=smpc, feature_block_size=size, separator="\t", report="", min_samples=0,
                    position=0, reference_batch=False, batch_labels=["site_a"],
                    feature_names=features, hash2feature=dict.fromkeys(features),
                    hash2variable={"covariate": "covariate_name"},
                    data=pd.DataFrame(np.ones((5, 4)), index=features),
                    design=pd.DataFrame(np.ones((4, 3))),
                )
                cohort.create_design.return_value = None
                cohort.compute_XtX_XtY.return_value = (xtx, xty, "")
                waiting = [False]

                def send(data, **kwargs):
                    if kwargs.get("memo"):
                        self.assertIs(kwargs["use_smpc"], smpc)
                        self.assertFalse(waiting[0], "Next block sent before awaiting the ACK")
                        waiting[0] = True

                def receive(**kwargs):
                    memo = kwargs["memo"]
                    if memo == "commonVariables":
                        return ["covariate"], 2
                    if memo == "commonGenes":
                        return features, protocol.clients
                    if memo == "beta":
                        return protocol.broadcast_data.call_args.args[0] if coordinator else reference
                    self.assertTrue(memo.startswith("block_done_"))
                    waiting[0] = False

                protocol.send_data_to_coordinator.side_effect = send
                protocol.await_data.side_effect = receive
                with patch.object(fedrbe, "Client", return_value=cohort), \
                        patch.object(fedrbe, "select_common_features_variables",
                                     return_value=(features, presence, protocol.clients)), \
                        tempfile.TemporaryDirectory() as output, \
                        contextlib.redirect_stdout(io.StringIO()):
                    if fail_later:
                        with self.assertRaisesRegex(ValueError, "Cannot reliably fit"):
                            fedrbe.run_fedrbe(protocol, "input", output)
                        cohort.remove_batch_effects.assert_not_called()
                        cohort.data_corrected.to_csv.assert_not_called()
                        cohort.full_corrected_data.to_csv.assert_not_called()
                        self.assertEqual(list(Path(output).iterdir()), [])
                        self.assertFalse(any(call.kwargs.get("memo") == "beta"
                                             for call in protocol.broadcast_data.call_args_list))
                    else:
                        fedrbe.run_fedrbe(protocol, "input", output)
                        np.testing.assert_array_equal(
                            cohort.remove_batch_effects.call_args.args[0], reference,
                        )
                        sends = [call.args[0] for call in protocol.send_data_to_coordinator.call_args_list
                                 if call.kwargs.get("memo")]
                        self.assertEqual([len(data[0]) for data in sends],
                                         [len(block[0]) for block in blocks])
                        np.testing.assert_array_equal(np.concatenate([data[0] for data in sends]), xtx)
                        np.testing.assert_array_equal(np.concatenate([data[1] for data in sends]), xty)
                for call in protocol.broadcast_data.call_args_list:
                    if call.kwargs.get("memo", "").startswith("block_done_"):
                        self.assertIsNone(call.args[0])

if __name__ == "__main__":
    unittest.main()
