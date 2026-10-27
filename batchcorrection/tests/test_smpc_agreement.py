"""Bounded state checks for SMPC agreement and the coordinator's coefficient broadcast."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from batchcorrection import classes
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
    }
    with patch.dict(sys.modules, aliases), patch.object(engine, "app", engine.App()):
        spec.loader.exec_module(states)
    return states, engine


class SmpcAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states, cls.engine = load_states()

    def make_state(self, state_class):
        state = state_class()
        state._app = self.engine.App()
        state._app.id = "site_a"
        state._app.coordinator = True
        state._app.clients = ["site_a", "site_b"]
        state.send_data_to_coordinator = Mock()
        state.broadcast_data = Mock()
        state.await_data = Mock(return_value=(["covariate"], 2))
        return state

    def make_initial_state(self, local_smpc, remote_smpc):
        state = self.make_state(self.states.InitialState)
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
        state.gather_data = Mock(return_value=[
            (["site_a"], ["covariate"], local_smpc),
            (["site_b"], ["covariate"], remote_smpc),
        ])
        return state, cohort

    def test_mixed_settings_stop_before_broadcast_or_feature_exchange(self):
        for local_smpc in (False, True):
            with self.subTest(coordinator_smpc=local_smpc):
                state, cohort = self.make_initial_state(local_smpc, not local_smpc)
                with patch.object(self.states, "Client", return_value=cohort):
                    with self.assertRaisesRegex(RuntimeError, "SMPC|smpc"):
                        state.run()
                state.broadcast_data.assert_not_called()

    def test_matching_boolean_settings_continue_initialization(self):
        for smpc in (False, True):
            with self.subTest(smpc=smpc):
                state, cohort = self.make_initial_state(smpc, smpc)
                with patch.object(self.states, "Client", return_value=cohort):
                    self.assertEqual(state.run(), "global_feature_selection")
                self.assertIs(
                    state.send_data_to_coordinator.call_args_list[0].args[0][-1], smpc,
                )

    def test_solver_receives_error_bound_from_smpc_settings(self):
        for smpc, exponent, expected in ((False, 12, 0.0), (True, 13, 2e-13)):
            with self.subTest(smpc=smpc, exponent=exponent):
                state = self.make_state(self.states.ComputeCorrectionState)
                state.configure_smpc(exponent=exponent)
                state.log = Mock()
                state.store("smpc", smpc)
                state.store("client", SimpleNamespace(
                    feature_names=["feature"], design=np.ones((4, 2))
                ))
                state.store("feature_presence_matrix", np.ones((1, 2)))
                state.gather_data = Mock(return_value=[])
                with patch.object(self.states, "compute_beta") as solve:
                    state.run()
                self.assertEqual(solve.call_args.kwargs["xtx_error"], expected)

    def test_only_batch_coefficients_are_broadcast(self):
        state = self.make_state(self.states.ComputeCorrectionState)
        state.configure_smpc(exponent=12)
        state.log = Mock()
        state.store("smpc", False)
        # Design columns: intercept, one covariate, one batch column (two batches).
        state.store("client", SimpleNamespace(
            feature_names=["feature"], design=np.ones((4, 3))
        ))
        state.store("feature_presence_matrix", np.ones((1, 2)))
        state.gather_data = Mock(return_value=[])
        with patch.object(self.states, "compute_beta", return_value=np.array([[1., 2., 3.]])):
            self.assertEqual(state.run(), "include_correction")
        np.testing.assert_array_equal(state.broadcast_data.call_args.args[0], [[3.]])
        self.assertEqual(state.broadcast_data.call_args.kwargs["memo"], "beta")


if __name__ == "__main__":
    unittest.main()
