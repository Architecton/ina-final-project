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
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from classifiers.feat_stacking_clf import FeatureStackingClf
from classifiers.gboostclf import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import time
import warnings
import argparse
from termcolor import colored


### PARSING ARGUMENTS ####################################################
def n_runs_check(val):
    ival = int(val)
    if ival <= 0:
        raise argparse.ArgumentTypeError('The specified number of runs must be greater than 0.')
    return ival
parser = argparse.ArgumentParser(description='Evaluate link prediction methods on specified network.')
parser.add_argument('networks', type=str, nargs='+')
parser.add_argument('--n-runs', type=n_runs_check, default=1, 
        help='Number of runs of the train-test split to perform.')
parser.add_argument('--clf', type=str, default='rf', 
        choices=['rf', 'svm', 'stacking', 'logreg', 'gboosting', 'majority', 'uniform'], help='Classifier to use for evaluation')
parser.add_argument('--scatter', type=str, nargs='+', 
        help='plot a scatterplot for up to three different features') # TODO
parser.add_argument('--eval-features', action='store_true', 
        help='Evaluate feature importance using specified classifier (if the classifier supports this)')
args = parser.parse_args()
if args.scatter is not None and len(args.scatter) not in (2, 3):
    parser.error('Can only plot 2 or 3 different features'.format(len(args.scatter)))
##########################################################################



### INITIALIZATION #######################################################

# Read list of features to extract.
with open('features.txt', 'r') as f:
    features_to_extract = list(filter(lambda x: x != '' and x[0] != '#', 
        map(lambda x: x.strip(), f.readlines())))


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
    SUBSET_SIZE = 30
    clf = FeatureStackingClf()
    clf.name = 'stacking'
elif args.clf == 'logreg':
    clf = LogisticRegression(max_iter=1000, C=1)
    clf.name = 'logreg'
elif args.clf == 'gboosting':
    clf = GradientBoostingClassifier() 
    clf.name = 'gboosting'
elif args.clf == 'majority':
    clf = DummyClassifier(strategy='most_frequent')
    clf.name = 'majority'
elif args.clf == 'uniform':
    clf = DummyClassifier(strategy='uniform')
    clf.name = 'uniform'

# Add classifier to pipeline.
clf_pipeline.steps.append(['clf', clf])

##########################################################################



### EVALUATION ###########################################################

# Go over specified networks and evaluate prediction method on each one.
for idx, network_path in enumerate(args.networks):

    print("evaluating {0} ({1}/{2})".format(network_path[network_path.rfind('/')+1:], idx+1, len(args.networks)))

    # Start timing evaluation.
    start_time = time.time()

    # Parse network and relabel nodes.
    network_raw = nx.read_edgelist(network_path, create_using=nx.Graph)
    network = nx.convert_node_labels_to_integers(network_raw, first_label=0)

    # Initialize feature extractor. 
    feature_extractor = features.get_feature_extractor(network, features_to_extract)

    # Set number of train-test split repetitions and initialize counter of repetitions.
    N_REP = args.n_runs
    idx_tts = 1

    # Initialize accumulators for classification accuracy, precision, recall, f-score and support.
    acc_aggr = 0
    prfs_aggr = np.zeros((4, 2), dtype=float)
        
    # Initialize list for evaluation scores (for Bayesian correlated t-test).
    scores_list = []

    # Perform specified number of train-test splits and average classification metrics.
    for train_edges_pos, test_edges_pos, train_edges_neg, test_edges_neg, test_network in train_test_split.repeated_train_test_split(network, n_rep=N_REP, test_size=0.2):

        # Get features for training and test nodes.
        features_train_pos = features.get_features(network, train_edges_pos, feature_extractor)
        features_train_neg = features.get_features(network, train_edges_neg, feature_extractor)
        features_test_pos = features.get_features(network, test_edges_pos, feature_extractor)
        features_test_neg = features.get_features(network, test_edges_neg, feature_extractor)

        # Format data (concatenate and shuffle)
        data_train, target_train = formatting.format_data(features_train_pos, features_train_neg, shuffle=True)
        data_test, target_test = formatting.format_data(features_test_pos, features_test_neg, shuffle=True)
        
        # If using Stacking method, set feature subset sizes.
        if clf_pipeline.named_steps['clf'].name == 'stacking':
            subset_lengths = [SUBSET_SIZE] * (data_train.shape[1]//SUBSET_SIZE) + [data_train.shape[1] % SUBSET_SIZE]
            clf_pipeline.named_steps['clf'].subset_lengths = subset_lengths

        # Train classifier and make predictions on test data.
        clf_pipeline.fit(data_train, target_train)
        pred = clf_pipeline.predict(data_test)
        
        # If evaluating feature importance and first network.
        if args.eval_features and idx_tts == 1:

            if len(args.networks) > 1:
                warnings.warn('Feature importance will be estimated only '
                        'on the first specified network ({0}).'.format(network_path[network_path.rfind('/')+1:]), RuntimeWarning)

            # If classifier supports feature scoring.
            if clf_pipeline.named_steps['clf'].name in {'gboosting', 'logreg'}:

                # Parse feature names.
                with open('./features.txt', 'r') as f:
                    feature_names = list(filter(lambda x: x != '' and x[0] != '#', 
                        map(lambda x: x.strip(), f.readlines())))

                    # Get dictionary mapping feature enumerations to feature names.
                    f_to_name = {'f' + str(idx) : feature_names[idx] for idx in range(len(feature_names))}

                # Get feature scores.
                if clf_pipeline.named_steps['clf'].name == 'gboosting':
                    scores = clf_pipeline.named_steps['clf'].score_features(f_to_name)
                elif clf_pipeline.named_steps['clf'].name == 'logreg':
                    weights = np.abs(clf_pipeline.named_steps['clf'].coef_[0])/sum(np.abs(clf_pipeline.named_steps['clf'].coef_[0]))
                    scores = {f_to_name['f' + str(idx)] : weights[idx] for idx in range(len(weights))}

                # Write features scores to file.
                with open('../results/feature_scores.txt', 'a') as f:
                    f.write('#####\n')
                    f.write('classifier: {0}\n'.format(clf_pipeline.named_steps['clf'].name))
                    f.write('network: {0}\n'.format(network_path[network_path.rfind('/')+1:]))
                    f.write('#####\n\n')
                    for (name, score) in [(name, score) for name, score in \
                            sorted(scores.items(), key=lambda x: x[1], reverse=True)]:
                        f.write('{0}: {1:.4f}\n'.format(name, score))
                    f.write('\n')
            else:
                warnings.warn('Feature importance estimation not supported for '
                        'selected classifier ({0}). Skipping.'.format(clf_pipeline.named_steps['clf'].name), 
                        RuntimeWarning)


        # Compute accuracy, precision, recall, f-score and support and add to accumulator.
        acc_aggr += sklearn.metrics.accuracy_score(target_test, pred)
        prfs_aggr += np.vstack(sklearn.metrics.precision_recall_fscore_support(target_test, pred))
        
        # Append score to list (for Bayesian correlated t-test).
        scores_list.append(sklearn.metrics.accuracy_score(target_test, pred))
        
        # If at last train-test split, plot confusion matrix and ROC curve.
        if idx_tts == N_REP:
            # Plot confusion matrix and ROC curve.
            results_processing.plot.confusion_matrix(data_test, target_test, clf_pipeline, clf_pipeline.named_steps['clf'].name, network_path[network_path.rfind('/')+1:])
            results_processing.plot.roc(data_test, target_test, clf_pipeline, clf_pipeline.named_steps['clf'].name, network_path[network_path.rfind('/')+1:])

        # Increment train-test split counter.
        idx_tts += 1
    
    # Get average accuracy, precision, recall, f-score and support.
    avg_prfs = prfs_aggr/N_REP
    avg_acc = acc_aggr/N_REP
    
    # Process and save results.
    results_processing.process.process_and_save(avg_acc, avg_prfs, N_REP, clf.name, network_path[network_path.rfind('/')+1:])

    # Save list of scores (for Bayesian correlated t-test).
    np.save('../results/statistical_tests/score_lists/' + network_path[network_path.rfind('/')+1:] + '_' + clf_pipeline.named_steps['clf'].name + '.npy', np.array(scores_list))
    
    # Compute evaluation running time.
    evaluation_runtime = time.time() - start_time

    # Print time needed for evaluation.
    print(colored('Evaluation took {0:.3f} seconds'.format(evaluation_runtime), 'red' if evaluation_runtime > 120.0 else 'green'))

        
##########################################################################

