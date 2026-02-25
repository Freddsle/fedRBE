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

    def run_test(self, retry: int = 0) -> Tuple[List[str], int, ExperimentResult]:
        """
        Runs the experiment. Requires `_startup` to be called beforehand with the same `data_dir`.
        Raises a `RuntimeError` on failure. Retries up to 5 times if the controller fails.

        Args:
            retry: Current retry attempt (int)

        Returns:
            Tuple:
            - result_files: List of result file paths (str)
            - coordinator_idx: Index of the coordinator's result file (int)
            - experiment_meta_info: Experiment metadata (`ExperimentResult` object)

        Raises:
            RuntimeError: If the experiment fails after 5 retries
        """
        if retry == 0:
            print("_______________EXPERIMENT_______________")

        ### Check prerequisites
        client = docker.from_env()
        try:
            client.images.get(self.app_image_name)
        except:
            raise RuntimeError(f"App image {self.app_image_name} not found!")

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

        ### Run the test
        exp_id, experiment_meta_info = self._start_test(retry=retry)
        # check_test will retry itself if needed, plus will only return a value once it's finished
        instances, dirs = self._check_test(exp_id=exp_id, retry=retry)
        print(f"instances: {instances}")
        print("TEST DONE:")
        ### Get the result file names
        result_files = list()
        instances.sort(key=lambda x: x['id']) # sort by id so that the order
                                              # is always the client order (client 0 at index 0)
        coord_idx = None
        for idx, info in enumerate(instances):
            try:
                if bool(info['coordinator']):
                    coord_idx = idx
            except:
                raise RuntimeError("Instance does not have a coordinator key! " +\
                                   "fc_controller API might have changed!")
            file_path = os.path.join(self.fc_data_dir,
                                     'tests',
                                     TESTS_DIR,
                                     f"results_test_{exp_id}_client_{info['id']}_{info['name']}.zip")
            result_files.append(file_path)

        if coord_idx is None:
            print("No coordinator found in the instances!")
            print("Restarting the test now!")
            self._startup()
            self.run_test(retry=retry+1)

        print("_______________EXPERIMENT FINISHED SUCCESSFULLY_______________")
        if coord_idx is None:
            raise RuntimeError("No coordinator found in the instances!")
        return result_files, coord_idx, experiment_meta_info

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

    def _start_test(self, retry: int = 0) -> Tuple[int, ExperimentResult]:
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
            print(f"{e}")
            retry += 1
            if retry > 5:
                raise RuntimeError(f"Test could not be started more than 5 times! Error: \n{e}")
            time.sleep(5)
            self._startup()
            return self._start_test(retry=retry)


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
        if self.generic_dir is None:
            self.generic_dir = "."
        elif self.generic_dir != ".":
            if os.path.exists(self.generic_dir) == False:
                raise RuntimeError(f"Generic directory {self.generic_dir} does not exist!")
            else:
                self.generic_dir = os.path.relpath(self.generic_dir, self.fc_data_dir)
        if self.channel is None:
            self.channel = "local"
        if self.query_interval is None:
            self.query_interval = 5

        client_dirs = ",".join([os.path.relpath(client, self.fc_data_dir) for client in self.clients])
        try:
            exp_id = fc.start(controller_host=self.controller_host,
                    app_image=self.app_image_name,
                    client_dirs=client_dirs,
                    generic_dir=self.generic_dir,
                    channel=self.channel,
                    query_interval=self.query_interval,
                    download_results=TESTS_DIR)
        except Exception as e:
            retry += 1
            if retry > 5:
                raise RuntimeError(f"Test could not be started more than 5 times! Error: \n{e}")
            print(f"Could not start the test! Error: \n{e}")
            print(f"retrying for the {retry}nth time")
            time.sleep(5)
            self._startup()
            return self._start_test(retry=retry)


        print("Test started successfully!")
        return int(exp_id), experiment_meta_info

    def _check_test(self, exp_id: int, retry: int = 0) -> Tuple[List[dict], List[str]]:
        """
        Given a test id, checks the status of the test until it is finished.
        If any errors occure, raises a RuntimeError.
        Retries automatically upto 5 times in case e.g. the controller dies.
        Retriying is done using the run_test function with incremented retry variable.
        Timeouts if self.timeout is exceeded. Retries again then.
        Args:
            exp_id (int): The number of the test to be checked
            retry (int): The number of current retries, if the test could not be started, the function calls the _startup
                        function and then itself with increased retry count.
                        If retry exceeds 5, the function raises a RuntimeError.
        Returns:
            Tuple of the following entries:
            instances (List[dict]): A list of dictionaries containing the instances (clients) of the test
                This information can be used to e.g. gather the result filenames
                in a further step.

        """
        ### Get the status of the experiment
        if self.controller_host is None:
            self.controller_host = DEFAULT_CONTROLLER_HOST
        time_run = int(time.time())
        while True:
            try:
                test_info = fc.info(controller_host=self.controller_host, test_id=exp_id)
            except Exception as e:
                print(f"Could not get the test info! Error: \n{e}")
                retry += 1
                print(f"retrying for the {retry}nth time")
                if retry > 5:
                    raise RuntimeError(f"Test could not be checked more than 5 times! Error: \n{e}")
                time.sleep(5)
                self._kill_leftover_container()
                self.run_test(retry=retry)

            status = test_info.iloc[0]['status']
            instances = test_info.iloc[0]['instances']
            dirs = test_info.iloc[0]['dirs']
            if status == "finished":
                print("Test finished successfully!")
                return instances, dirs
            if status == "error" or status == "stopped":
                raise RuntimeError(f"Test finished with an error or was stopped! Status: {status}")
            # timeout handling
            if int(time.time()) - time_run > self.timeout:
                print("Test timed out! Restarting it now!")
                fc.stop(controller_host=self.controller_host, test_id=exp_id)
                time.sleep(30) # wait for the controller to stop the test
                retry += 1
                self.run_test(retry=retry)
            # in any other case we just continue until the test exists with an
            # error or is finished
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
        Eliminates any containers left of the app image
        """
        print("Killing leftover containers...")
        dockerclient = docker.from_env()
        containers = dockerclient.containers.list()

        # Iterate through the containers
        image_name = self.app_image_name
        if "/" in image_name:
            image_name = image_name.split("/")[-1]
        for container in containers:
            if image_name in container.name:
                # Stop the container
                try:
                    container.stop()
                except Exception as e:
                    print(f"WARNING: failed to stop a container, will try to continue the workflow. Error: {e}")
                print(f"Stopped leftover container: {container.id} running image {self.app_image_name}")
                time.sleep(5) # just to make sure the container is really removed

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
