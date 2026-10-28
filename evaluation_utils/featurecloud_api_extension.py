# pylint: disable=all
"""
This file serves to automatically run a featureCloud app with different
configurations and input data.
It uses three basic building blocks for this:
1. _startup() function
    This function starts the featurecloud controller with the given data directory
    All data used in the runs that want to be done should be in the data directory.
2. Experiment() class
    This class is used to define an experiment, where an experiment is the
    execution of one app with one specific configuration.
    See the class for more details
2. run_test() function
    This function runs the given experiment with the given data directory.
    It returns a list of the result files that can be used to further analyse the results.
    Resultfiles are zip files of the results that depend on the app used.
    Performs multiple sanity checks and is capable of restarting the controller
    if needed. Will retry upto 5 times before raising an error.
"""

from dataclasses import dataclass
from typing import List, Union, Dict, Tuple
import subprocess
import os
import time
import hashlib
import yaml
import docker
import json
from copy import deepcopy

from FeatureCloud.api.imp.test import commands as fc
from FeatureCloud.api.imp.controller import commands as fc_controller

DEFAULT_CONTROLLER_HOST = "http://localhost:8000"
TESTS_DIR = 'tests'
MAX_RETRIES = 5


class _RetryableTestError(RuntimeError):
    """A failed attempt that requires a new controller test."""


### Classes
@dataclass
class ExperimentResult():
    """
    This dataclass is used to define the result of an experiment.
    Each attribute represents a column in the result dataframe.
    """
    experiment_name: str
        # The name of the experiment, taken from the Experiment class
    input_hashes: str
        # A JSON string in the format:
        # {"client_0": {"filename": "hash"}, ..., "generic": {"filename: "hash"}}
        # if the file cannot be hashed the hash is None
    config: str
        # A JSON string of the used config file

@dataclass
class Experiment():
    """
    This dataclass is used to define all circumstances of one
    featurecloud test run. It is used together with the run_test function.
    """
    name: str  # The name of the experiment to it identifiable
    clients: List[str]  # List of client folders (absolute paths)
    app_image_name: str # The name of the app image to be used,
                        # expects the latest tag to be used!
    fc_data_dir: str    # The data directory to be used by the featurecloud controller
    config_files: Union[List[dict], None] = None
                        # List of all config files to be used as given as dicts,
                        # order is the same as clients
                        # if None, no config files are used, so either
                        # no config file should be necessary or the client folders
                        # containg the correct config.yaml files already
    config_file_changes: Union[List[dict], None] = None
                        # List of all config file changes.
                        # order is the same as clients.
                        # if a nested key is changed, . is used as
                        # separator. Overwrites/writes the default
                        # config given in config_files
                        # example would be the entry {"key1.key2": "value"}
                        # then the test would be run with the base config
                        # but base_config[key1][key2] would be set to value
                        # if None, no changes are applied
    generic_dir: Union[str, None] = None    # The generic folder (absolute path)
                                            # it's content is added to all clients!
    channel: Union[str, None] = None # The channel to be used,
                                     # if None, the default channel is used!
                                     # Options are local and internet
    query_interval: Union[int, None] = None  # The query interval in seconds,
                                             # None for default
    controller_host: Union[str, None] = None # The controller host to be used
                                    # if None, DEFAULT_CONTROLLER_HOST is used
    timeout: int = 900  # The timeout in seconds for the test. if a test takes
                        # longer than this, it is stopped and retried
                        # default is 15 minutes

    def run_test(self) -> Tuple[List[str], int, ExperimentResult]:
        """
        Runs the experiment. Requires `_startup` to be called beforehand with the same `data_dir`.
        Retries up to five times after test launch failures, unavailable test status,
        timeouts, or no reported coordinator. Each retry launches a fresh test.
        A missing coordinator field is a metadata error and is not retried.

        Returns:
            Tuple:
            - result_files: List of result file paths (str)
            - coordinator_idx: Index of the coordinator's result file (int)
            - experiment_meta_info: Experiment metadata (`ExperimentResult` object)

        Raises:
            RuntimeError: If the experiment fails after 5 retries
        """
        if self.query_interval is not None and self.query_interval <= 0:
            raise ValueError("query_interval must be positive")
        print("_______________EXPERIMENT_______________")

        ### Check prerequisites
        client = docker.from_env()
        try:
            client.images.get(self.app_image_name)
        except:
            raise RuntimeError(f"App image {self.app_image_name} not found! Use docker pull to receive it!")

        # Ensure the controller is up
        try:
            self._kill_leftover_container()
            controller_instance = fc_controller.status(name=fc_controller.DEFAULT_CONTROLLER_NAME)
            if controller_instance.status != 'running':
                print("Controller not running, starting it now!")
                self._startup()
        except:
            print("Controller not running, starting it now!")
            self._startup()

        for attempt in range(MAX_RETRIES + 1):
            try:
                exp_id, experiment_meta_info = self._start_test()
                instances = self._check_test(exp_id=exp_id)
                # Client IDs determine the order of the returned archive paths.
                instances.sort(key=lambda info: info['id'])
                coord_idx = self._coordinator_index(instances)
                result_files = [
                    os.path.join(
                        self.fc_data_dir, 'tests', TESTS_DIR,
                        f"results_test_{exp_id}_client_{info['id']}_{info['name']}.zip",
                    )
                    for info in instances
                ]
            except _RetryableTestError as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Experiment failed after {MAX_RETRIES} retries: {exc}"
                    ) from exc
                print(f"{exc}\nRetrying ({attempt + 1}/{MAX_RETRIES}) with a new test.")
                time.sleep(5)
                self._startup()
                continue

            print(f"instances: {instances}")
            print("_______________EXPERIMENT FINISHED SUCCESSFULLY_______________")
            return result_files, coord_idx, experiment_meta_info

    def _coordinator_index(self, instances: List[dict]) -> int:
        """Find the coordinator in client order, requiring complete metadata."""
        coord_idx = None
        for idx, info in enumerate(instances):
            if 'coordinator' not in info:
                raise RuntimeError(
                    "Instance does not have a coordinator key! "
                    "fc_controller API might have changed!"
                )
            if bool(info['coordinator']):
                coord_idx = idx
        if coord_idx is None:
            raise _RetryableTestError("No coordinator found in the instances!")
        return coord_idx

    ### Helpers
    def _startup(self) -> None:
        """
        Starts the featurecloud controller with the given data- directory
        If any errors occure, raises a RuntimeError.
        """
        ### Check prerequisites
        # docker
        try:
            subprocess.run(["docker", "ps"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except:
            raise RuntimeError("Docker is not installed or not running!")
        # data_dir exists
        if not os.path.exists(self.fc_data_dir):
            raise RuntimeError(f"Data directory {self.fc_data_dir} given for the _startup does not exist!")

        ### Start the featurecloud controller
        try:
            self._kill_leftover_container()
            fc_controller.stop(name=fc_controller.DEFAULT_CONTROLLER_NAME)
            time.sleep(5)
        except Exception as e:
            raise RuntimeError(f"Could not stop the controller as needed for a clean start: {e}") from e
        print(f"Starting the FeatureCloud controller with the data directory {self.fc_data_dir}")
        try:
            if ".." in self.fc_data_dir:
                raise RuntimeError(f"Data directory should not contain .. ! data-dir: {self.fc_data_dir}")
            fc_controller.start(name=fc_controller.DEFAULT_CONTROLLER_NAME,
                                port=8000, data_dir=self.fc_data_dir,
                                controller_image="",
                                with_gpu=False, mount="",
                                blockchain_address="")
        except Exception as e:
            raise RuntimeError(f"Could not start the controller! Error: {e}") from e

        print("Controller started successfully, waiting 5s to ensure the controller is properly running!")
        time.sleep(5)

    def _start_test(self) -> Tuple[int, ExperimentResult]:
        """
        Starts a test with the given experiment configuration.
        Applies the experiment.config_file_changes given to experiment.config_files,
        saves the config files as config.yaml in the client folders and starts the test.
        If any errors occure, raises a RuntimeError.
        Overwrites any config.yaml files already present in the client folders.
        Returns:
            A tuple (test_id, experiment_meta_info) with the following entries:
                test_id (int): the number of the test started, can be used to check the test
                experiment_meta_info (ExperimentResult): An ExperimentResult instance
                    containing the meta information of the experiment
        """
        ### Check prerequisites
        if self.config_files is None and self.config_file_changes is not None:
            raise RuntimeError("If config_file_changes are given, config_files should be given as well!")
        if self.config_files:
            if len(self.config_files) != len(self.clients):
                raise RuntimeError("The number of clients and config files should be the same!")
            if self.config_file_changes:
                if len(self.config_file_changes) != len(self.clients):
                    raise RuntimeError("The number of clients and config file changes should be the same!")
        for client in self.clients:
            if not os.path.exists(client):
                raise RuntimeError(f"Client folder {client} does not exist!")

        ### Get the correct config files to be used
        self._set_config_files()

        ### Fill the experiment meta info to be returned
        try:
            input_hashes = self._get_input_hashes()
        except Exception as e:
            raise _RetryableTestError(f"Could not hash the test inputs: {e}") from e


        # finally fill the ExperimentResult instance
        experiment_meta_info = ExperimentResult(experiment_name=self.name,
                                                input_hashes=json.dumps(input_hashes),
                                                config=json.dumps(self.config_files))

        ## finally start the test
        # we need to set the default values in case some experiment attribute
        # is None
        if self.clients is None:
            raise RuntimeError("The clients should be given!")
        if self.app_image_name is None:
            raise RuntimeError("The app image name should be given!")
        if self.controller_host is None:
            self.controller_host = DEFAULT_CONTROLLER_HOST
        generic_dir = self.generic_dir or ""
        if generic_dir not in ("", "."):
            if os.path.exists(self.generic_dir) == False:
                raise RuntimeError(f"Generic directory {self.generic_dir} does not exist!")
            else:
                generic_dir = os.path.relpath(self.generic_dir, self.fc_data_dir)
        if self.channel is None:
            self.channel = "local"
        if self.query_interval is None:
            self.query_interval = 5

        client_dirs = ",".join([os.path.relpath(client, self.fc_data_dir) for client in self.clients])
        try:
            exp_id = fc.start(controller_host=self.controller_host,
                    app_image=self.app_image_name,
                    client_dirs=client_dirs,
                    generic_dir=generic_dir,
                    channel=self.channel,
                    query_interval=self.query_interval,
                    download_results=TESTS_DIR)
        except Exception as e:
            raise _RetryableTestError(f"Could not start the test: {e}") from e


        print("Test started successfully!")
        return int(exp_id), experiment_meta_info

    def _check_test(self, exp_id: int) -> List[dict]:
        """
        Given a test id, checks the status of the test until it is finished.
        Polls one started attempt. Status failures and timeouts request a retry from
        run_test, which owns the retry budget and the replacement test metadata.
        Args:
            exp_id (int): The number of the test to be checked
        Returns:
            Client instance metadata used to locate result archives.
        """
        ### Get the status of the experiment
        time_run = time.monotonic()
        while True:
            try:
                test_info = fc.info(controller_host=self.controller_host, test_id=exp_id)
            except Exception as e:
                raise _RetryableTestError(
                    f"Could not get info for test {exp_id}: {e}"
                ) from e

            status = test_info.iloc[0]['status']
            if status == "finished":
                print("Test finished successfully!")
                return test_info.iloc[0]['instances']
            if status == "error" or status == "stopped":
                raise RuntimeError(
                    f"Test {exp_id} finished with an error or was stopped! Status: {status}. "
                    f"Inspect app/controller logs in {os.path.join(self.fc_data_dir, 'logs')}."
                )
            # timeout handling
            if time.monotonic() - time_run > self.timeout:
                fc.stop(controller_host=self.controller_host, test_id=exp_id)
                time.sleep(30) # wait for the controller to stop the test
                raise _RetryableTestError(f"Test {exp_id} timed out after {self.timeout}s")
            time.sleep(self.query_interval)

    def _get_input_hashes(self):
        input_hashes: Dict[str, Dict[str, Union[str, None]]] = dict()
            # see ExperimentResult class for more details

        # Get all input files in the client folders
        for idx, client_dir in enumerate(sorted(self.clients)):
            input_hashes[f"client_{idx}"] = dict()
            for root, _, files in os.walk(client_dir):
                for file in files:
                    # sometimes we get some weird files that do not really
                    # exist, on my machine it's usually
                    # ../../.mozilla/firefox/<some version>.default-release/lock,
                    # so a firefox lock file
                    # We should not be in any .. case at all, so we raise an
                    # error if the filepath contains ..
                    file_path = os.path.join(root, file)
                    if ".." in file_path:
                        raise RuntimeError(f"ERROR: os.walk got lost in .. :(")
                    if file_path in input_hashes[f"client_{idx}"]:
                        raise RuntimeError(f"File {file_path} exists twice!")
                    try:
                        file_hash = hash_file(file_path)
                        input_hashes[f"client_{idx}"][file_path] = file_hash
                    except FileNotFoundError:
                        print(f"WARNING: File {file_path} does not exist!")
                        print(f"Skipping this file!")
                        continue
                    except Exception as e:
                        raise RuntimeError(f"ERROR: Could not hash file {file_path}! Error: {e}")

        # get the files in the generic directory
        input_hashes["generic"] = dict()
        if self.generic_dir is not None:
            for root, _, files in os.walk(self.generic_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path in input_hashes["generic"]:
                        raise RuntimeError(f"File {file_path} exists twice!")
                    file_hash = hash_file(file_path)
                    input_hashes["generic"][file_path] = file_hash

        return input_hashes


    def _set_config_files(self):
        """
        Applies the changes given in self.config_file_changes to
        self.config_files and saved the corresponding files in the
        client folders as config.yaml
        """
        if self.config_files:
            # we deepcopoy the config files to not change the original
            # config files. We deepcopy in the calling script as well, but
            # just to be sure we do it here as well, in case the calling
            # script does not do it.
            self.config_files = [deepcopy(config) for config in self.config_files]
            if self.config_file_changes:
                for config_changes, config in zip(self.config_file_changes, self.config_files):
                    # Make sure flimmaBatchCorrection exists in the base config
                    if "flimmaBatchCorrection" not in config:
                        raise RuntimeError("Base config does not have 'flimmaBatchCorrection' section.")

                    # Extract changes intended for flimmaBatchCorrection
                    flimma_changes = config_changes.get("flimmaBatchCorrection", {})

                    # Update only the flimmaBatchCorrection section
                    for key, value in flimma_changes.items():
                        config["flimmaBatchCorrection"][key] = value

        if not self.config_files:
            raise RuntimeError("No config files given!")
        ### os manipulations to get the config files in the correct client folders
        for client_dir, config in zip(self.clients, self.config_files):
            config_path = os.path.join(client_dir, "config.yml")
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

    def _kill_leftover_container(self):
        """
        Eliminates any containers left of the app image.
        Matches by image tag because FeatureCloud strips underscores from
        image names when generating container names (e.g. fc_kmeans_upd ->
        fc_fckmeansupd_*), so matching by container.name is unreliable.
        Uses remove(force=True) to both stop and remove containers so they
        are fully disconnected from Docker networks.
        """
        print("Killing leftover containers...")
        dockerclient = docker.from_env()
        # include stopped containers — they stay on networks until removed
        containers = dockerclient.containers.list(all=True)

        image_name = self.app_image_name
        if "/" in image_name:
            image_name = image_name.split("/")[-1]
        # FC strips underscores when naming containers, so also try without them
        image_name_no_underscores = image_name.replace("_", "")

        for container in containers:
            tags = container.image.tags if container.image else []
            name_match = (
                image_name in container.name
                or image_name_no_underscores in container.name
            )
            image_match = any(image_name in tag for tag in tags)
            if name_match or image_match:
                try:
                    container.remove(force=True)  # stops + removes in one call
                    print(f"Removed leftover container: {container.id} (image: {self.app_image_name})")
                except Exception as e:
                    print(f"WARNING: failed to remove container {container.id}, will try to continue. Error: {e}")

### Helper functions
def hash_file(file_path, exclude=None):
    """Generate a SHA-256 hash of the contents of a file."""
    hasher = hashlib.sha512()
    if exclude is None:
        exclude = []
    if os.path.basename(file_path) in exclude:
        return None
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
