import numpy as np
import networkx as nx
from eval_utils import train_test_split
from feature_extraction import features, formatting
import results_processing.plot
import results_processing.process
import sklearn.metrics
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from classifiers.feat_stacking_clf import FeatureStackingClf
from sklearn.pipeline import Pipeline
import argparse


### PARSING ARGUMENTS ####################################################
def n_runs_check(val):
    ival = int(val)
    if ival <= 0:
        raise argparse.ArgumentTypeError('The specified number of runs must be greater than 0.')
    return ival
parser = argparse.ArgumentParser(description='Evaluate link prediction methods on specified network.')
parser.add_argument('networks', type=str, nargs='+')
parser.add_argument('--n-runs', type=n_runs_check, default=1)
parser.add_argument('--clf', type=str, default='rf', 
        choices=['rf', 'svm', 'stacking'], help='classifier to use for evaluation')
args = parser.parse_args()
##########################################################################


### INITIALIZATION #######################################################
# Read list of features to extract.
with open('features.txt', 'r') as f:
    features_to_extract = list(filter(lambda x: x != '' and x[0] != '#', map(lambda x: x.strip(), f.readlines())))

# Get feature extractor.
if features_to_extract is not None:
    feature_extractor = features.get_feature_extractor(features_to_extract)
else:
    raise FileNotFoundError('features.txt file containing names of features to extract not found')


# Initialize pipeline template.
clf_pipeline = Pipeline([('scaling', RobustScaler())])


# Initialize classifier.
if args.clf == 'rf':
    clf = RandomForestClassifier(n_estimators=100)
    clf.name = 'rf'
elif args.clf == 'svm':
    clf = SVC(gamma='auto', probability=True)
    clf.name = 'svm'
elif args.clf == 'stacking':
    # TODO TODO TODO
    subset_lengths = [2, 2, 2]
    clf = FeatureStackingClf()
    clf.name = 'stacking'

# Add classifier to pipeline.
clf_pipeline.steps.append(['clf', clf])

##########################################################################



### EVALUATION ###########################################################

# Go over specified networks and evaluate prediction method on each one.
for network_name in args.networks:

    # Parse network.
    DATA_PATH = '../data/' + network_name
    network = nx.read_edgelist(DATA_PATH, create_using=nx.Graph)

    # Set number of train-test split repetitions and initialize counter of repetitions.
    N_REP = args.n_runs
    idx_tts = 1

    # Initialize accumulators for classification accuracy, precision, recall, f-score and support.
    acc_aggr = 0
    prfs_aggr = np.zeros((4, 2), dtype=float)

    # Perform specified number of train-test splits and average classification metrics.
    for train_edges_pos, test_edges_pos, train_edges_neg, test_edges_neg, test_network in train_test_split.repeated_train_test_split(network, n_rep=N_REP, test_size=0.1):
        
        # Get features for training and test nodes.
        features_train_pos = features.get_features(network, train_edges_pos, feature_extractor)
        features_train_neg = features.get_features(network, train_edges_neg, feature_extractor)
        features_test_pos = features.get_features(network, test_edges_pos, feature_extractor)
        features_test_neg = features.get_features(network, test_edges_neg, feature_extractor)

        # Format data (concatenate and shuffle)
        data_train, target_train = formatting.format_data(features_train_pos, features_train_neg)
        data_test, target_test = formatting.format_data(features_test_pos, features_test_neg)

        # Train classifier and make predictions on test data.
        clf.fit(data_train, target_train)
        pred = clf.predict(data_test)
        
        # Compute accuracy, precision, recall, f-score and support and add to accumulator.
        acc_aggr += sklearn.metrics.accuracy_score(target_test, pred)
        prfs_aggr += np.vstack(sklearn.metrics.precision_recall_fscore_support(target_test, pred))
        
        # If at last train-test split, plot confusion matrix and ROC curve.
        if idx_tts == N_REP:
            # Plot confusion matrix and ROC curve.
            results_processing.plot.confusion_matrix(data_test, target_test, clf, network_name)
            results_processing.plot.roc(data_test, target_test, clf, network_name)

        # Increment train-test split counter.
        idx_tts += 1
    
    # Get average accuracy, precision, recall, f-score and support.
    avg_prfs = prfs_aggr/N_REP
    avg_acc = acc_aggr/N_REP
    
    # Process and save results.
    results_processing.process.process_and_save(avg_acc, avg_prfs, N_REP, clf.name, network_name)

##########################################################################

