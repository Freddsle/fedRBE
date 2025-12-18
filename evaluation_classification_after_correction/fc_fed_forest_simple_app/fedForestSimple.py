
from typing import Optional
import os
import yaml
import pandas as pd
import numpy as np
from math import ceil
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
    # Expected files in inputfolder:
    - config_forest.yaml:
    ```
    simple_forest:
        data_filename: "data.csv"
            # the file containing data, see further down as another expected input
        csv_seperator: "\t"
        train_test_ratio: 1.0
            # ratio of samples to take as training data
            # makes sure that training and test data both contain all to be predicted
        predicted_feature_name: "cancer"
        num_estimators_total: 5
        max_depth: 10
        min_samples_split: 2
        min_samples_leaf: 1
        max_features: 'sqrt'
        random_state: 42
        features_as_columns: True # whether features are given as columns or rows
        sample_col: 0
            # Ignored if features_as_columns = False
            # Which column contains the sample indexes. If not given each row is considered an
            # individual and independent sample
        feature_name_col: 0
            # Ignored if features_as_columns = True
            # Which col contains the feature names. By default the first column.
    - <data.csv>: The file containing the data to be used
    # Resulting output in outputfolder:
    - global_metrics.csv: A simple csv with the following columns:
        [num_samples_training, num_samples_test, num_classes, metric, score]
        The following metrics are tested: MCC, F1_score
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

    # Load data
    data_path = os.path.join(inputfolder, config['data_filename'])
    separator = config.get('csv_seperator', ',')

    df = pd.read_csv(data_path, sep=separator)

    # Prepare data based on orientation
    if config.get('features_as_columns', True):
        # Features are columns
        sample_col = config.get('sample_col', None)
        if sample_col is not None:
            df = df.set_index(df.columns[sample_col])

        # Separate features and target
        target_col = config['predicted_feature_name']
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        # Features are rows
        feature_name_col = config.get('feature_name_col', 0)
        df = df.set_index(df.columns[feature_name_col])
        df = df.T  # Transpose so samples are rows

        target_col = config['predicted_feature_name']
        y = df[target_col]
        X = df.drop(columns=[target_col])

    # Ensure numeric features
    X = X.apply(pd.to_numeric, errors='coerce')

    # Train-test split with stratification
    train_ratio = config.get('train_test_ratio', 0.8)
    random_state = config.get('random_state', 42)

    if train_ratio >= 1.0:
        #print("Using all data for training (no test split).")
        X_train = X
        y_train = y
        X_test = X
        y_test = y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_ratio,
        random_state=random_state,
        stratify=y
        )

    # Get number of clients (this would come from the protocol)
    # For now, we'll get it after gathering data
    num_clients = None

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

        #print(f"Coordinator: Found {len(common_features)} common features across {len(all_feature_names)} clients")

        # Broadcast common features
        protocol_fed_learning.broadcast_data(common_features, send_to_self=True)

    # All clients (including coordinator) receive common features
    common_features = protocol_fed_learning.await_data(n=1, unwrap=True)

    # Filter datasets to only include common features
    X_train = X_train[common_features]
    X_test = X_test[common_features]

    # Calculate trees per client
    num_clients = len(protocol_fed_learning.clients)
    trees_per_client = ceil(num_estimators_total / num_clients)

    # Phase 1: Train local decision trees
    local_trees = []
    for i in range(trees_per_client):
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state + i if random_state else None
        )

        tree.fit(X_train, y_train)
        local_trees.append(tree)

    # Phase 2: Send local trees to coordinator
    protocol_fed_learning.send_data_to_coordinator(local_trees, send_to_self=True)

    if protocol_fed_learning.is_coordinator:
        # Gather all trees from clients
        all_trees_data = protocol_fed_learning.gather_data(is_json=False)

        # Combine all trees into a single ensemble
        all_trees = []
        for trees_data in all_trees_data:
            all_trees.extend(trees_data)

        #print(f"Coordinator: Received {len(all_trees)} trees total from {len(all_trees_data)} clients")

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

    # Phase 3: Calculate local confusion matrix counts and share with coordinator
    y_pred_test = global_forest.predict(X_test)

    # Get unique classes for binary/multiclass support
    classes = np.unique(y)
    num_classes = len(classes)

    # Calculate aggregated TP, TN, FP, FN across all classes (one-vs-rest)
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for cls in classes:
        # Convert to binary: current class vs rest
        y_test_binary = (y_test == cls).astype(int)
            # is this entry equal to the class?
        y_pred_binary = (y_pred_test == cls).astype(int)
            # is this entry predicted as the class?

        total_tp += int(np.sum((y_test_binary == 1) & (y_pred_binary == 1)))
        total_tn += int(np.sum((y_test_binary == 0) & (y_pred_binary == 0)))
        total_fp += int(np.sum((y_test_binary == 0) & (y_pred_binary == 1)))
        total_fn += int(np.sum((y_test_binary == 1) & (y_pred_binary == 0)))

    # Create local metrics with aggregated counts
    local_confusion_matrix = {
        'num_samples_training': len(X_train),
        'num_samples_test': len(X_test),
        'num_classes': num_classes,
        'TP': total_tp,
        'TN': total_tn,
        'FP': total_fp,
        'FN': total_fn
    }

    # store the local metrics
    mcc_numerator = (total_tp * total_tn) - (total_fp * total_fn)
    mcc_denominator = np.sqrt((total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn))
    local_mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0.0
    local_f1_numerator = 2 * total_tp
    local_f1_denominator = 2 * total_tp + total_fp + total_fn
    local_f1 = local_f1_numerator / local_f1_denominator if local_f1_denominator != 0 else np.nan

    local_metrics_data = [
        {
            'num_samples_training': len(X_train),
            'num_samples_test': len(X_test),
            'num_classes': num_classes,
            'class': 'local_client',
            'metric': 'MCC',
            'score': local_mcc,
        },
        {
            'num_samples_training': len(X_train),
            'num_samples_test': len(X_test),
            'num_classes': num_classes,
            'class': 'local_client',
            'metric': 'F1_score',
            'score': local_f1,
        }
    ]

    # Save local metrics
    local_metrics_df = pd.DataFrame(data=local_metrics_data)
    local_metrics_path = os.path.join(outputfolder, "local_metrics.csv")
    local_metrics_df.to_csv(local_metrics_path, index=False)

    # Send local metrics to coordinator
    protocol_fed_learning.send_data_to_coordinator(local_confusion_matrix, send_to_self=True)

    if protocol_fed_learning.is_coordinator:
        # Gather all local metrics from clients
        all_local_metrics = protocol_fed_learning.gather_data()

        # Aggregate counts globally
        global_num_samples_training = sum(m['num_samples_training'] for m in all_local_metrics)
        global_num_samples_test = sum(m['num_samples_test'] for m in all_local_metrics)
        num_classes = all_local_metrics[0]['num_classes']

        # Aggregate TP, TN, FP, FN from all clients
        total_tp = sum(m['TP'] for m in all_local_metrics)
        total_tn = sum(m['TN'] for m in all_local_metrics)
        total_fp = sum(m['FP'] for m in all_local_metrics)
        total_fn = sum(m['FN'] for m in all_local_metrics)

        # Calculate global MCC
        # https://medium.com/@anishnama20/matthews-correlation-coefficient-mcc-one-of-the-best-metric-when-2-classes-are-imbalanced-c0318ac68c21
        numerator = (total_tp * total_tn) - (total_fp * total_fn)
        denominator = np.sqrt((total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn))
        mcc = numerator / denominator if denominator != 0 else 0.0

        # Calculate unweighted (micro-averaged) F1 score
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        numerator = 2 * total_tp
        denominator = 2 * total_tp + total_fp + total_fn
        f1 = numerator / denominator if denominator != 0 else 0.0

        # Create metrics data
        global_metrics_data = [
            {
                'num_samples_training': global_num_samples_training,
                'num_samples_test': global_num_samples_test,
                'num_classes': num_classes,
                'class': 'overall',
                'metric': 'MCC',
                'score': mcc,
            },
            {
                'num_samples_training': global_num_samples_training,
                'num_samples_test': global_num_samples_test,
                'num_classes': num_classes,
                'class': 'overall',
                'metric': 'F1_score',
                'score': f1,
            }
        ]

        # Save metrics (only coordinator)
        metrics_df = pd.DataFrame(data=global_metrics_data)

        metrics_path = os.path.join(outputfolder, "global_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

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


    # Save predictions (all clients save their local predictions)
    # Combine train and test data with predictions

    # Train predictions
    X_train_pred = X_train.copy()
    X_train_pred[config['predicted_feature_name']] = y_train
    X_train_pred['prediction_simple_forest'] = global_forest.predict(X_train)
    X_train_pred['dataset'] = 'train'

    # Test predictions
    if train_ratio < 1.0:
        X_test_pred = X_test.copy()
        X_test_pred[config['predicted_feature_name']] = y_test
        X_test_pred['prediction_simple_forest'] = y_pred_test
        X_test_pred['dataset'] = 'test'

        # Combine train and test
        pred_df = pd.concat([X_train_pred, X_test_pred], axis=0)
    else:
        pred_df = X_train_pred

    pred_path = os.path.join(outputfolder, "pred.csv")
    pred_df.to_csv(pred_path, sep=separator)
