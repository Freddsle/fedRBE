"""
This file serves to automatically run a featureCloud app with different
configurations and input data. 
It uses three basic building blocks for this:
1. startup(data_dir: str) -> None
    This function starts the featurecloud controller with the given data directory
    All data used in the runs that want to be done should be in the data directory.
2. Experiment() class
    This class is used to define an experiment, where an experiment is the
    execution of one app with one specific configuration.
    See the class for more details
2. run_test(experiment: Experiment, data_dir: str, retry: int = 0) -> List[str]
    This function runs the given experiment with the given data directory.
    It returns a list of the result files that can be used to further analyse the results.
    Resultfiles are zip files of the results that depend on the app used.
    Performs multiple sanity checks and is capable of restarting the controller
    if needed. Will retry upto 5 times before raising an error.

"""

from dataclasses import dataclass
from typing import List, Union
import subprocess
import os
import yaml
import docker
import time

from FeatureCloud.api.imp.test import commands as fc
from FeatureCloud.api.imp.controller import commands as fc_controller

### Classes
@dataclass
class Experiment():
    clients: List[str]  # List of client folders (absolute paths)
    app_image_name: str # The name of the app image to be used,  
                        # expects the latest tag to be used! 
    config_files: List[dict] # List of all config files to be used as given as dicts, 
                             # order is the same as clients
    config_file_changes: List[dict] # List of all config file changes. 
                                     # order is the same as clients.
                                     # if a nested key is changed, . is used as
                                     # separator. Overwrites/writes the default 
                                     # config given in config_files
                                     # example would be the entry {"key1.key2": "value"}
                                     # then the test would be run with the base config
                                     # but base_config[key1][key2] would be set to value
    generic_dir: Union[str, None] = None    # The generic folder (absolute path)
                                            # it's content is added to all clients!
    channel: Union[str, None] = None # The channel to be used, if None, the default channel is used!
                                     # Options are local and internet
    query_interval: Union[int, None] = None  # The query interval in seconds, None for default
    controller_host: Union[str, None] = None # The controller host address, None for default (http://localhost:8000)

### Functions
def startup(data_dir: str) -> None:
    """
    Starts the featurecloud controller with the given data- directory 
    If any errors occure, raises a RuntimeError.
    Args:
        data_dir (str): The directory to be used mounted as the data folder
                        should contain all clients files, FeatureCloud does not
                        have access to any other files! Should be given as an absolute directory.
    """
    ### Check prerequisites
    # docker
    try:
        subprocess.run(["docker", "ps"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except:
        raise RuntimeError("Docker is not installed or not running!")
    # data_dir exists
    if os.path.exists(data_dir) == False:
        raise RuntimeError(f"Data directory {data_dir} given for the startup does not exist!")

    ### Start the featurecloud controller
    try:
        fc_controller.stop(name="")
    except Exception as e:
        raise RuntimeError(f"Could not stop the controller as needed for a clean start: {e}")
    print(f"Starting the FeatureCloud controller with the data directory {data_dir}")
    try:
        fc_controller.start(name="fc-controller", port=8000, data_dir=data_dir, controller_image="", with_gpu=False, mount="", blockchain_address="")
    except Exception as e:
        raise RuntimeError(f"Could not start the controller! Error: {e}")
    
    print("Controller started successfully, waiting 5s to ensure the controller is properly running!")
    time.sleep(5)
    
def run_test(experiment: Experiment, data_dir: str, retry: int = 0) -> List[str]:
    """
    Runs the experiment given. Prerequisite is that startup has been called before
    using the same data_dir. If any errors occure, raises a RuntimeError.
    Retries automatically upto 5 times in case e.g. the controller dies.
    Args:
        experiment (Experiment): The experiment to be run, see the Experiment class for more details
        data_dir (str): The data_dir used by the controller. Needed if the controller has to be restarted.
        retry (int): The number of retries, if the test could not be started, the function calls the startup
                     function and then itself with increased retry count. 
                     If retry exceeds 5, the function raises a RuntimeError.
    """
    if retry == 0:
        print("_______________EXPERIMENT_______________")
    
    ### Check prerequisites
    client = docker.from_env()
    try:
        client.images.get(experiment.app_image_name)
    except:
        raise RuntimeError(f"App image {experiment.app_image_name} not found!")

    ### Run the test
    exp_id = start_test(experiment=experiment, data_dir=data_dir, retry=retry)
    instances = check_test(exp_id=exp_id, experiment=experiment, data_dir=data_dir, retry=retry)

    ### Get the result file names
    result_files = list()
    for info in instances:
        result_files.append(f"results_test_{exp_id}_client_{info['id']}_{info['name']}.zip")
    return result_files


def start_test(experiment: Experiment, data_dir: str, retry: int = 0, ) -> int:
    """
    Starts a test with the given experiment configuration. 
    Applies the experiment.config_file_changes given to experiment.config_files,
    saves the config files as config.yaml in the client folders and starts the test.
    If any errors occure, raises a RuntimeError.
    Overwrites any config.yaml files already present in the client folders.
    Args:
        experiment (Experiment): The experiment to be started, see the Experiment class for more details
        data_dir (str): The data_dir used by the controller. Needed if the controller has to be restarted.
        retry (int): The number of retries, if the test could not be started, the function calls the startup
                     function and then itself with increased retry count. 
                     If retry exceeds 5, the function raises a RuntimeError.
    Returns:
        Returns the number of the test started, can be used to check the test status
    """
    ### Check prerequisites
    if len(experiment.clients) != len(experiment.config_files) != len(experiment.config_file_changes):
        raise RuntimeError("The number of clients, config files and config file changes should be the same!")
    for client in experiment.clients:
        if os.path.exists(client) == False:
            raise RuntimeError(f"Client folder {client} does not exist!")
        
    ### Get the correct config files to be used
    for config_changes, config in zip(experiment.config_file_changes, experiment.config_files):
        for key, value in config_changes.items():
            if "." in key:
                keys = key.split(".")
                config_pointer = config
                for k in keys[:-1]:
                    if k not in config_pointer:
                        raise RuntimeError(f"Key {k} to be modified not found in the config file {config}!")
                    config_pointer = config[k]
                if keys[-1] not in config_pointer:
                    raise RuntimeError(f"Key {keys[-1]} to be modified not found in the config file {config}!")
                config_pointer[keys[-1]] = value
            else:
                config[key] = value
    
    ### os manipulations to get the config files in the correct client folders
    for client_dir, config in zip(experiment.clients, experiment.config_files):
        config_path = os.path.join(client_dir, "config.yaml") 
        with open(config_path, "w") as f:
            print(f"Writing the config file to {config_path}")
            yaml.dump(config, f, default_flow_style=False)

    ## finally start the test
    # we need to set the default values in case some experiment attribute
    # is None
    if experiment.clients is None:
        raise RuntimeError("The clients should be given!")
    if experiment.app_image_name is None:
        raise RuntimeError("The app image name should be given!")
    if experiment.controller_host is None:
        experiment.controller_host = "http://localhost:8000"
    if experiment.generic_dir is None:
        experiment.generic_dir = "."
    else:
        if os.path.exists(experiment.generic_dir) == False:
            raise RuntimeError(f"Generic directory {experiment.generic_dir} does not exist!")
        else:
            experiment.generic_dir = os.path.relpath(experiment.generic_dir, data_dir)
    if experiment.channel is None:
        experiment.channel = "local"
    if experiment.query_interval is None:
        experiment.query_interval = 5

    client_dirs = ",".join([os.path.relpath(client, data_dir) for client in experiment.clients])
    try:
        exp_id = fc.start(controller_host=experiment.controller_host,
                 app_image=experiment.app_image_name,
                 client_dirs=client_dirs,
                 generic_dir=experiment.generic_dir,
                 channel=experiment.channel,
                 query_interval=experiment.query_interval,
                 download_results="tests")
    except Exception as e:
        retry += 1
        if retry > 5:
            raise RuntimeError(f"Test could not be started more than 5 times! Error: \n{e}")
        print(f"Could not start the test! Error: \n{e}")
        print(f"retrying for the {retry}nth time")
        time.sleep(5)
        startup(data_dir=data_dir)
        # check if there are some leftover containers
        # Get a list of all containers
        dockerclient = docker.from_env()
        containers = dockerclient.containers.list()

        # Iterate through the containers
        for container in containers:
            # Check if the container is running the specified image
            if experiment.app_image_name in container.image.tags:
                # Stop the container
                container.stop()
                print(f"Stopped leftover container: {container.id} running image {experiment.app_image_name}")
        return int(start_test(experiment=experiment, data_dir=data_dir, retry=retry))

    
    print("Test started successfully!")
    return int(exp_id)


def check_test(exp_id: int, experiment: Experiment, data_dir: str, retry: int) -> List[dict]:
    """
    Given a test id, checks the status of the test until it is finished.
    If any errors occure, raises a RuntimeError.
    Retries automatically upto 5 times in case e.g. the controller dies.
    Retriying is done using the run_test function with incremented retry variable.
    Args:
        exp_id (int): The id of the test to be checked
        experiment (Experiment): The experiment that was run
        data_dir (str): The data_dir used by the controller. Needed if the controller has to be restarted.
        retry (int): The number of current retries, if the test could not be started, the function calls the startup
                     function and then itself with increased retry count. 
                     If retry exceeds 5, the function raises a RuntimeError.
    Returns:
        instances (List[dict]): A list of dictionaries containing the instances (clients) of the test
            This information can be used to e.g. gather the result filenames 
            in a further step.
    """
    ### Get the status of the experiment
    if experiment.controller_host is None:
        experiment.controller_host = "http://localhost:8000"
    while True:
        try:
            test_info = fc.info(controller_host=experiment.controller_host, test_id=exp_id)
        except Exception as e:
            print(f"Could not get the test info! Error: \n{e}")
            retry += 1
            print(f"retrying for the {retry}nth time")
            if retry > 5:
                raise RuntimeError(f"Test could not be checked more than 5 times! Error: \n{e}")
            time.sleep(5)
            run_test(experiment=experiment, data_dir=data_dir, retry=retry)
        
        status = test_info.iloc[0]['status']
        instances = test_info.iloc[0]['instances']
        if status == "finished":
            print("Test finished successfully!")
            return instances
            break
        elif status == "error" or status == "stopped":
            raise RuntimeError(f"Test finished with an error or was stopped! Status: {status}")
        # in any other case we just continue until the test exists with an
        # error or is finished