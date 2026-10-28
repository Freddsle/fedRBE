"""
Runs a simulation of any federated algorithm implemented via the ProtocolFedLearning
natively (without docker), one thread per client.
Usage:
Call run_simulation_native with the correct parameters.
Your federated learning algorithm should be implemented as described by the fed_learning_main_function
of run_simulation_native.
"""
from typing import List, Callable, Optional

import threading
from .localfedlearningsimulator import LocalFedLearningSimulationWrapper
from .protocolfedlearning import ProtocolFedLearning

def run_simulation_native(clientpaths: List[str], outputfolders: List[str], generic_dir: Optional[str],
                          fed_learning_main_function: Callable[[ProtocolFedLearning, Optional[str], Optional[str]], None],
                          client_ids: Optional[List] = None):
    """
    Using the helpers from localfedlearningsimulator.py, run the simulation natively.

    Args:
        clientpaths: The path to the clientfolders. Recommendation to use absolute paths.
        outputfolders: The path to the outputfolders. Recommendation to use absolute paths.
            The first clientpath and first outputfolder are used as the coordinator, the client
            usually storing global models and metrics.
        generic_dir: The path to the generic directory. The content of this folder is copied to
            each client folder. If the file in the generic folder already exists in the
            client folder, it is not copied. Folders in the generic folder are not copied.
            The first clientfolder is used as the coordinator.
        client_ids: The IDs of the clients in the order of clientpaths, available
            via the protocol's id and clients. Defaults to 0, ..., number of clients-1.
    """
    # create the wrapper
    wrapper = LocalFedLearningSimulationWrapper(clientfolders=clientpaths,
                                                outputfolders=outputfolders,
                                                generic_dir=generic_dir,
                                                client_ids=client_ids)

    # Preserve the first failure before waking peers blocked on an exchange.
    failures = []
    failure_lock = threading.Lock()

    def run_client(local_client, inputfolder, outputfolder):
        try:
            fed_learning_main_function(local_client, inputfolder, outputfolder)
        except BaseException as error:
            with failure_lock:
                failures.append(error)
                wrapper.shared_dict.abort_event.set()

    threads = []
    try:
        for idx, local_client in enumerate(wrapper.clients):
            thread = threading.Thread(
                target=run_client,
                args=(local_client, clientpaths[idx], outputfolders[idx]))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    finally:
        wrapper.shared_dict.abort_event.set()
        for thread in threads:
            thread.join()
        wrapper.cleanup_created_files()
    if failures:
        raise failures[0]
