import os

from FeatureCloud.app.engine.app import AppState, app_state

from fedrbe import run_fedrbe, SMPC_EXPONENT

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
# The whole protocol is implemented in fedrbe.run_fedrbe, which uses this
# AppState for sending and receiving data.
INPUT_FOLDER = os.path.join(os.getcwd(), "mnt", "input")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "mnt", "output")
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        self.configure_smpc(exponent=SMPC_EXPONENT) # set the default values, we use a bigger exponent though
        run_fedrbe(self, INPUT_FOLDER, OUTPUT_FOLDER)
        return 'terminal'
