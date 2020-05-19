import numpy as np
from bayesiantests.bayesiantests import correlated_ttest, correlated_ttest_MC
import matplotlib.pyplot as plt
import seaborn as snb
import argparse


def make_test(data1, data2, name1, name2, xlabel, network_name, rope=0.01):
    """
    Perform Bayesian Correlated t-test, save probabilities to file
    and save plot.

    Args:
        data1 (numpy.ndarray): Results of performing 10 runs of 10-fold CV using first classifier
        data2 (numpy.ndarray): Results of performing 10 runs of 10-fold CV using second classifier
        name1 (str): Name of first classifier
        name2 (str): Name of second classifier
        xlabel (str): x-axis label for the plot
        network_name (str): Network name on which the classifiers were evaluated
        rope (float): Region of practical equivalence
    """
    
    # Get names and make data matrix.
    names = (name1, name2)
    data_mat = np.empty((len(data1), 2), dtype=float)
    data_mat[:, 0] = data1
    data_mat[:, 1] = data2
    
    # Generate samples from posterior (it is not necesssary because the posterior is a Student).
    samples = correlated_ttest_MC(data_mat, rope=rope, runs=1, nsamples=50000)
    l, w, r = correlated_ttest(data_mat, rope=rope, runs=1, verbose=True, names=names)

    # Plot posterior.
    snb.kdeplot(samples, shade=True)

    # Plot rope region.
    plt.axvline(x=-rope, color='orange')
    plt.axvline(x=rope, color='orange')
    
    # Set x-axis label.
    plt.xlabel(xlabel)
    plt.savefig('./plots/bctt_' + network_name.replace('-', '_') + '_' + \
            name1.replace(' ', '_').lower() + '_' + name2.replace(' ', '_').lower() +  '.png')

    # Save probabilities to file.
    with open('./results_bctt.txt', 'a') as f:
        f.write('#####\n')
        f.write(network_name)
        f.write('\n#####\n\n')
        f.write('p({0} > {1})={2}, '.format(name1, name2, l))
        f.write('p({0})={1}, '.format('EQ', w))
        f.write('p({0} > {1})={2}\n\n'.format(name2, name1, r))


if __name__ == '__main__':

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data1_path', type=str, required=True, help='Path to file containing CV results for first classifier')
    parser.add_argument('--data2_path', type=str, required=True, help='Path to file containing CV results for second classifier')
    parser.add_argument('--network-name', type=str, required=True, help='Name of network on which the classifiers were evaluated')
    parser.add_argument('--xlabel', type=str, required=True, help='x-axis label for the plot')
    parser.add_argument('--name1', type=str, required=True, help='Name of first classifier')
    parser.add_argument('--name2', type=str, required=True, help='Name of second classifier')
    args = parser.parse_args()
    
    # Load CV results (10 runs of 10-fold cross-validation).
    data1 = np.load(args.data1_path)
    data2 = np.load(args.data2_path)
    
    # Perform Bayesian Correlated t-test and store results.
    make_test(data1, data2, args.name1, args.name2, args.xlabel, args.network_name, rope=0.01)


