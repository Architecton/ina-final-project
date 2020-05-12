import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def confusion_matrix(data_test, target_test, clf, network_name):
    """
    Plot and save confuction matrix.

    Args:
        data_test (numpy.ndarray): Test data samples
        target_test (numpy.ndarray): Target labels (target variable values)
        clf (object): Trained classifier for which to plot the confusion matrix.
        network_name (str): The name of the network being evaluated.
    """

    # Set print precision.
    np.set_printoptions(precision=2)

    # Plot confusion matrix and save plot.
    disp = metrics.plot_confusion_matrix(clf, data_test, target_test,
                                 display_labels=['no link', 'link'],
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')

    disp.figure_.set_size_inches(9.0, 9.0, forward=True)
    plt.tight_layout()
    plt.savefig('../results/plots/cfm_' + network_name + '_' + clf.name + '.png')
    plt.clf()
    plt.close()


def roc(data_test, target_test, clf, network_name):
    """
    Plot ROC curve using train-test split and save results to file.
    
    Args:
        data_test (numpy.ndarray): Test data samples
        target_test (numpy.ndarray): Target labels (target variable values)
        clf (object): Trained classifier for which to plot the confusion matrix.
        network_name (str): The name of the network being evaluated.
    """
    
    # Predict probabilities.
    scores = clf.predict_proba(data_test)
    
    # Get false positive rates, true positive rates and thresholds.
    fpr, tpr, thresholds = metrics.roc_curve(target_test, scores[:, 1], pos_label=1)

    # Compute AUC.
    roc_auc = metrics.roc_auc_score(target_test, scores[:, 1])
    
    # Plot ROC curve. 
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = {0:4f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('../results/plots/roc_' + network_name + '_' + clf.name + '.png')
    plt.clf()
    plt.close()

