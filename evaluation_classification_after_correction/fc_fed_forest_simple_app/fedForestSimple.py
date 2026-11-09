
from typing import Optional
import os
import yaml
import pandas as pd
import numpy as np
from math import ceil
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import validate_data
import pickle
import sklearn

from .fc_fedlearnsim_helper.protocolfedlearning import ProtocolFedLearning


def main(protocol_fed_learning: ProtocolFedLearning,
         inputfolder: Optional[str] = None,
         outputfolder: Optional[str] = None) -> None:
    """
    Implements a simple federated random forst algorithm for classification.
    Creates ceil(num_estimators/clients) decision trees per client, aggregates them globally
    and creates as the final global model simply the ensemble of all clients trees.
    All given samples are used for training, test data must be split off before
    and evaluated with the saved global model.
    # Expected files in inputfolder:
    - config_forest.yaml:
    ```
    simple_forest:
        data_filename: "data.csv"
            # the file containing data, see further down as another expected input
        csv_seperator: "\t"
        predicted_feature_name: "cancer"
        num_estimators_total: 5
        max_depth: 10
        min_samples_split: 2
        min_samples_leaf: 1
        max_features: 'sqrt'
        random_state: 42
        features_as_columns: True # whether features are given as columns or rows
        save_predictions: True # Export per-sample features and predictions of the training data
        sample_col: 0
            # Ignored if features_as_columns = False
            # Which column contains the sample indexes. If not given each row is considered an
            # individual and independent sample
        feature_name_col: 0
            # Ignored if features_as_columns = True
            # Which col contains the feature names. By default the first column.
    - <data.csv>: The file containing the data to be used
    # Resulting output in outputfolder:
    - global_model.pkl, model_info.yaml, model_description.txt (coordinator only):
        The global forest and the features it uses
    - <pred.csv>: The given data in the format of columns being features.
        An additional prediction_simple_forest column is added.
    ```
    """
    # Default paths
    if inputfolder is None:
        inputfolder = "/mnt/input"
    if outputfolder is None:
        outputfolder = "/mnt/output"

    # Load configuration
    config_path = os.path.join(inputfolder, "config_forest.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['simple_forest']
    if 'train_test_ratio' in config:
        raise ValueError("train_test_ratio is not supported, the forest is trained on all given "
                         "samples. Split off the test data before.")

    # Load data
    data_path = os.path.join(inputfolder, config['data_filename'])
    separator = config.get('csv_seperator', ',')

    # read index in as column, later we will set it as index if needed based on config
    index_col = config.get('sample_col', None) if config.get('features_as_columns', True) else config.get('feature_name_col', 0)
    df = pd.read_csv(data_path, sep=separator, index_col=index_col)

    # Prepare data based on orientation
    if config.get('features_as_columns', True):
        # Separate features and target
        target_col = config['predicted_feature_name']
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        df = df.T  # Transpose so samples are rows
        target_col = config['predicted_feature_name']
        y = df[target_col]
        X = df.drop(columns=[target_col])

    # CSV numeric columns already have the required values and dtypes.
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        X = X.apply(pd.to_numeric, errors='coerce')

    # All given samples are training samples
    X_train = X
    y_train = y
    random_state = config.get('random_state', 42)

    # Extract forest parameters
    num_estimators_total = config.get('num_estimators_total', 100)
    max_depth = config.get('max_depth', None)
    min_samples_split = config.get('min_samples_split', 2)
    min_samples_leaf = config.get('min_samples_leaf', 1)
    max_features = config.get('max_features', 'sqrt')


    # Phase 0: Align features across clients
    # Send local feature names to coordinator
    local_feature_names = X_train.columns.tolist()
    protocol_fed_learning.send_data_to_coordinator(local_feature_names, send_to_self=True)

    if protocol_fed_learning.is_coordinator:
        # Gather all feature names from clients
        all_feature_names = protocol_fed_learning.gather_data(is_json=True)

        # Compute intersection of all feature sets
        common_features = set(all_feature_names[0])
        for feature_names in all_feature_names[1:]:
            common_features = common_features.intersection(set(feature_names))

        # Convert to sorted list for consistent ordering
        common_features = sorted(list(common_features))


        # Broadcast common features
        protocol_fed_learning.broadcast_data(common_features, send_to_self=True)

    # All clients (including coordinator) receive common features
    common_features = protocol_fed_learning.await_data(n=1, unwrap=True)

    # Filter datasets to only include common features
    X_train = X_train[common_features]

    # Calculate trees per client
    num_clients = len(protocol_fed_learning.clients)
    trees_per_client = ceil(num_estimators_total / num_clients)

    # Apply the estimator's dtype and feature-name validation once. This also
    # preserves its conversion rules for mixed numeric and boolean columns.
    input_validator = DecisionTreeClassifier()
    train_values = validate_data(
        input_validator, X_train, dtype=np.float32,
        accept_sparse="csc", ensure_all_finite=False)
    feature_names = getattr(input_validator, "feature_names_in_", None)

    # Phase 1: Train local decision trees
    local_trees = []
    # A run owns its tree RNG stream; adjacent run seeds must not reuse seed blocks.
    tree_seeds = np.random.RandomState(random_state).randint(
        np.iinfo(np.int32).max, size=trees_per_client)
    for tree_seed in tree_seeds:
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=int(tree_seed)
        )

        tree.fit(train_values, y_train)
        if feature_names is not None:
            tree.feature_names_in_ = feature_names
        local_trees.append(tree)
    del train_values

    # Phase 2: Send local trees to coordinator
    protocol_fed_learning.send_data_to_coordinator(local_trees, send_to_self=True)

    if protocol_fed_learning.is_coordinator:
        # Gather all trees from clients
        all_trees_data = protocol_fed_learning.gather_data(is_json=False)

        # Combine all trees into a single ensemble
        all_trees = []
        for trees_data in all_trees_data:
            all_trees.extend(trees_data)


        # Create global random forest with proper attributes
        global_forest = RandomForestClassifier(n_estimators=len(all_trees))
        global_forest.estimators_ = all_trees
        global_forest.n_classes_ = len(np.unique(y_train))
        global_forest.classes_ = np.unique(y_train)
        global_forest.n_features_in_ = X_train.shape[1]
        global_forest.n_outputs_ = 1
        global_forest.feature_names_in_ = X_train.columns.to_numpy()

        # Broadcast global forest
        protocol_fed_learning.broadcast_data(pickle.dumps(global_forest), send_to_self=True)

    # All clients (including coordinator) receive global forest
    global_forest_data = protocol_fed_learning.await_data(n=1, unwrap=True)
    global_forest = pickle.loads(global_forest_data) #type: ignore
        # we just sent a pickle.dumps object, linters complain but interpreters are fine

    if protocol_fed_learning.is_coordinator:
        # Save the global model and relevant information
        model_path = os.path.join(outputfolder, "global_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(global_forest, f)
        model_description_path = os.path.join(outputfolder, "model_description.txt")
        with open(model_description_path, 'w') as f:
            f.write(f"Global Random Forest with {len(global_forest.estimators_)} trees\n")
            f.write("To load the model, use:\n")
            f.write("import pickle\n")
            f.write("import yaml\n")
            f.write("model_info = yaml.safe_load(open('model_info.yaml', 'r'))\n")
            f.write("with open('global_model.pkl', 'rb') as f:\n")
            f.write("    global_forest = pickle.load(f)\n")
            f.write("predictions = global_forest.predict(X_test[model_info['used_features']])\n")
            f.write("Model is based on from sklearn.ensemble import RandomForestClassifier\n")
            f.write("from sklearn.tree import DecisionTreeClassifier\n")
            f.write(f"sklearn version used: {sklearn.__version__}\n")

        model_info_path = os.path.join(outputfolder, "model_info.yaml")
        with open(model_info_path, 'w') as f:
            yaml.dump({
                'num_trees': len(global_forest.estimators_),
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'random_state': random_state,
                'used_features': common_features
            }, f)


    if not config.get("save_predictions", True):
        return

    # Save the predictions of the training data (all clients save their local predictions)
    pred_df = X_train.copy()
    pred_df[config['predicted_feature_name']] = y_train
    pred_df['prediction_simple_forest'] = global_forest.predict(X_train)

    pred_path = os.path.join(outputfolder, "pred.csv")
    pred_df.to_csv(pred_path, sep=separator)
