"""
Contains two functions to run a simulation of any federated algorithm implemented via the
ProtocolFedLearning:
1. Via FeatureCloud's Testing module
2. Via a custom simulation (natively without docker)
Usage:
Simply call either run_simulation_featurecloud or run_simulation_native with the correct parameters.
Your federated learning algorithm should be implemented as described by the fed_learning_main_function
of run_simulation_native.
"""
from typing import List, Callable, Optional, Tuple

# featurecloud imports
import time
import docker
import FeatureCloud.api.imp.controller.commands as controller
import FeatureCloud.api.imp.test.commands as test

# native imports
import threading
from .localfedlearningsimulator import LocalFedLearningSimulationWrapper
from .protocolfedlearning import ProtocolFedLearning

def run_simulation_featurecloud(data_path: str, clientnames: List[str], generic_dir: str,
                                app_name: str):
    """
    Run the simulation via FeatureCloud's Testing module.
    Must be run from the root folder of the FeatureCloud app to be tested.

    Args:
        data_path: The path to the data generally. All clients data is in that folder
        clientnames: The path to the clientfolders from the data_path.
        generic_dir: The path to the generic directory. The content of this folder is used
            in all clients.
    """
    # build the image
    docker_client = docker.from_env()
    try:
        docker_client.images.build(path='.', tag=f'{app_name}:latest')
    except Exception as e:
        raise Exception(f"Could not build the FeatureCloud App image, make sure this function is called from the root folder of a FetureCloud App project: {e}") from e

    # stop and start the controller
    controller.stop(name='')
    print(f'Starting controller with data_dir={data_path}')
    controller.start(name=controller.DEFAULT_CONTROLLER_NAME,
                        port=8000,
                        data_dir=data_path,
                        controller_image='',
                        with_gpu=False,
                        mount='',
                        blockchain_address='')
    # wait some time until the controller is online
    time.sleep(10)

    # start the test
    test.start(controller_host='http://localhost:8000',
                client_dirs=','.join(clientnames),
                generic_dir=generic_dir,
                app_image=app_name,
                channel='local',
                query_interval=3,
                download_results='')

    # inform the user where they can see the test
    print('You can follow the test along at https://featurecloud.ai/development/test')

def run_simulation_native(clientpaths: List[str], outputfolders: List[str], generic_dir: Optional[str],
                          fed_learning_main_function: Callable[Tuple[ProtocolFedLearning, Optional[str], Optional[str]], Tuple[None]]):
    """
    Using the helpers from src.helper.localfedlearningsimulator.py, run the simulation natively.

    Args:
        clientpaths: The path to the clientfolders. Recommendation to use absolute paths.
        outputfolders: The path to the outputfolders. Recommendation to use absolute paths.
            The first clientpath and first outputfolder are used as the coordinator, the client
            usually storing global models and metrics.
        generic_dir: The path to the generic directory. The content of this folder is copied to
            each client folder. If the file in the generic folder already exists in the
            client folder, it is not copied. Folders in the generic folder are not copied.
            The first clientfolder is used as the coordinator.
    """
    # create the wrapper
    wrapper = LocalFedLearningSimulationWrapper(clientfolders=clientpaths,
                                                outputfolders=outputfolders,
                                                generic_dir=generic_dir)

    # run the simulation
    # start all clients as threads
    threads = []
    for idx, local_client in enumerate(wrapper.clients):
        threads.append(threading.Thread(
            target=fed_learning_main_function,
            args=(local_client, clientpaths[idx], outputfolders[idx])))
        threads[-1].start()

    # done, perform cleanup
    for thread in threads:
        thread.join()
    wrapper.cleanup_created_files()
