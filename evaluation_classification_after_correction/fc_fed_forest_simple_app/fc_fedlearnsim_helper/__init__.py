# TODO: desc of this package, basically only the run funcs are needed
'''
Purpose
This package serves as a helper for federated learning simulation using the FeatureCloud framework.
It allows to:
1. Programatically run a FeatureCloud federated learning test run given local folders, where
each folder represents one client
2. Given the FeatureCloud App is developed as described in the Usage section, you can with one
single function call run the federated learning simulation using multithreading natively.
Especially for multiple runs, this speeds up federated learning simulations considerably.

Usage
Develop your federated learning algorithm as one singular function that uses an instance of
ProtocolFedLearning to send and receive data as needed for federated learning.
This function should use the following parameters:
def fed_learning_main_function(protocol_fed_learning: ProtocolFedLearning,
         inputfolder: Optional[str] = None,
         outputfolder: Optional[str] = None):

You can then easily run this function via FeatureCloud or locally simulated with multi threading:
- FeatureCloud: In states.py, run this function, passing the AppState instance
(which implements ProtocolFedLearning) to the function. Sending and receiving data will then
happen via FeatureCloud. You can use run_simulation_featurecloud to run the app using local folders
via the FeatureCloud testembed
- Local simulation: You can simply use the run_simulation_native function.
This starts each client as a seperate thread, using a sharedDictionary between them for
sending data.
'''
