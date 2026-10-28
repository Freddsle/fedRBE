"""Bounded protocol checks for SMPC agreement and the coordinator's coefficient broadcast."""

import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from batchcorrection import classes, fedrbe
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
                (["site_a"], ["covariate"], local_smpc),
                (["site_b"], ["covariate"], remote_smpc),
            ]),
        )
        cohort = SimpleNamespace(
            smpc=local_smpc,
            separator="\t",
            hash2feature={"feature": "feature_name"},
            hash2variable={"covariate": "covariate_name"},
            batch_labels=["site_a"],
            min_samples=0,
            position=0,
            reference_batch=False,
            config_based_init=Mock(),
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
                    protocol.send_data_to_coordinator.call_args_list[0].args[0][-1], smpc,
                )
                protocol.broadcast_data.assert_called_once()

    def test_solver_error_bound_follows_smpc_settings(self):
        for smpc, exponent, expected in ((False, 12, 0.0), (True, 13, 2e-13)):
            with self.subTest(smpc=smpc, exponent=exponent):
                with patch.object(fedrbe, "SMPC_EXPONENT", exponent):
                    self.assertEqual(fedrbe.smpc_error_bound(smpc, 2), expected)

    def test_initial_state_runs_the_protocol_with_the_app_exponent(self):
        states, engine = load_states()
        state = states.InitialState()
        state._app = engine.App()
        with patch.object(states, "run_fedrbe") as run:
            self.assertEqual(state.run(), "terminal")
        run.assert_called_once_with(state, states.INPUT_FOLDER, states.OUTPUT_FOLDER)
        self.assertEqual(state._app.default_smpc["exponent"], fedrbe.SMPC_EXPONENT)

    def test_only_batch_coefficients_are_broadcast(self):
        cohorts = ["site_a", "site_b"]
        protocol = SimpleNamespace(
            id="site_a",
            is_coordinator=True,
            clients=cohorts,
            send_data_to_coordinator=Mock(),
            broadcast_data=Mock(),
            gather_data=Mock(side_effect=[
                [(["site_a"], ["covariate"], False), (["site_b"], ["covariate"], False)],
                [],  # feature information, consumed by the patched feature selection
                [],  # XtX and XtY, consumed by the patched solver
            ]),
            await_data=Mock(side_effect=[
                (["covariate"], 2), (["feature"], cohorts), np.array([[3.]]),
            ]),
        )
        cohort = MagicMock(
            smpc=False, separator="\t", report="", min_samples=0, position=0,
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
                patch.object(fedrbe, "compute_beta", return_value=np.array([[1., 2., 3.]])), \
                tempfile.TemporaryDirectory() as output:
            fedrbe.run_fedrbe(protocol, "input", output)
        beta_broadcast = protocol.broadcast_data.call_args
        self.assertEqual(beta_broadcast.kwargs["memo"], "beta")
        np.testing.assert_array_equal(beta_broadcast.args[0], [[3.]])


if __name__ == "__main__":
    unittest.main()
