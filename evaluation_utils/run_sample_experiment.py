import os
import requests
import time

import FeatureCloud.api.imp.test.commands as fc_test
import FeatureCloud.api.imp.controller.commands as controller
import FeatureCloud.api.imp.app.commands as fc_app

script_path = os.path.dirname(os.path.abspath(__file__)) # path of this script
data_path = os.path.join(os.path.dirname(script_path), "evaluation_data", "simulated", "mild_imbalanced", "before")
client_paths = 'lab1,lab2,lab3'

# ensure the controller is started with the correct folder
# first stop it
controller.stop("")

# then start it
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
    raise Exception(f"Failed to start the featurecloud controller with the correct folder: {e}")

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
