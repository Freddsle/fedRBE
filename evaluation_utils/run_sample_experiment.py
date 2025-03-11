import os
import subprocess
import requests
import time

import FeatureCloud.api.imp.test.commands as fc_test
import FeatureCloud.api.imp.controller.commands as controller
import FeatureCloud.api.imp.app.commands as fc_app

def stop_all_bcorrect_containers():
    """
    Stops all Docker containers running the image 'featurecloud.ai/bcorrect:latest'.
    """
    try:
        # Get the list of container IDs for the specified image.
        container_ids_output = subprocess.check_output(
            ["docker", "ps", "-q", "--filter", "ancestor=featurecloud.ai/bcorrect:latest"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        container_ids = container_ids_output.splitlines()

        if container_ids:
            # Stop the containers.
            subprocess.run(["docker", "stop"] + container_ids, check=True)
            print("Stopped containers:", container_ids)
        else:
            print("No containers found for featurecloud.ai/bcorrect:latest.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to stop bcorrect containers: {e.output.decode().strip()}")
    except Exception as e:
        raise Exception(f"Unexpected error when stopping bcorrect containers: {e}")

# Stop all containers using the specified image.
stop_all_bcorrect_containers()

script_path = os.path.dirname(os.path.abspath(__file__))  # path of this script
data_path = os.path.join(os.path.dirname(script_path), "evaluation_data", "simulated", "mild_imbalanced", "before")
client_paths = 'lab1,lab2,lab3'

# Ensure the controller is started with the correct folder.
# First stop it.
controller.stop("")

# Then start it.
try:
    controller.start(
        name=controller.DEFAULT_CONTROLLER_NAME,
        port=8000,
        data_dir=data_path,
        mount="",
        with_gpu=False,
        blockchain_address="",
        controller_image=""
    )
except Exception as e:
    raise Exception(f"Failed to start the FeatureCloud controller with the correct folder: {e}")

# wait for the controller to be online
time_passed = 0
print("(Re)starting the featurecloud controller with the correct folder")
while True:
    try:
        response = requests.get("http://localhost:8000")
        if response.status_code == 200:
            break
    except Exception as e:
        pass

    if time_passed > 60:
        print(time)
        raise Exception("Controller did not start in time")
    time_passed += 5
    time.sleep(5)

# we simply run the Microbiome experiment as sample data
print("Starting the experiment")
fc_test.start(
    controller_host='http://localhost:8000',
    client_dirs=client_paths,
    generic_dir="",
    app_image="featurecloud.ai/bcorrect:latest",
    channel="local",
    query_interval=5,
    download_results=""
)
print("Experiment started successfully")
print("You can follow the process of the experiment in the browser: https://featurecloud.ai/development/test")