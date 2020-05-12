import datetime


def process_and_save(avg_acc, avg_prfs, n_rep, clf_name, network_name):
    """
    Process and save classification analysis results in 
    the form of a markdown table.

    Args:
        avg_acc (float): The average accuracy for the runs.
        avg_prfs (numpy.ndarray): Average precission, recall,
        f-score and support.
        n_rep (int): Number of repetitions of the train-test
        split and evaluation performed.
        clf_name (str): Name of the classifier used.
        network_name (str): Name of the network on which the
        evaluation was performed.
    """

    with open('../results/results.txt', 'a') as f:

        # Write header.
        f.write('##########\n')
        f.write('Date: {0}\n'.format(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
        f.write('network: {0}\n'.format(network_name))
        f.write('clf: {0}\n'.format(clf_name))
        f.write('##########\n\n')
        f.write('average classification accuracy: {0:.4f}\n'.format(avg_acc))
        f.write('number of runs: {0}\n\n'.format(n_rep))

        # Expand last column to match variable number of digits in the support.
        expand0 = len(str(avg_prfs[-1, 0])) - 6
        expand1 = len(str(avg_prfs[-1, 1])) - 6
        match0 = expand1 - expand0
        match1 = expand0 - expand1
        expand_max = max(expand0, expand1)
        support0_f = '{0:.2f}'.format(avg_prfs[-1, 0]) + match0*' '
        support1_f = '{0:.2f}'.format(avg_prfs[-1, 1]) + match1*' '
        
        # Write table.
        f.write('| class   | precission | recall | f-score | support ' + expand_max*' ' + '|\n')
        f.write('|---------|------------|--------|---------|---------' + expand_max*'-' + '|\n')
        f.write('| {0} | {1:.2f}       | {2:.2f}   | {3:.2f}    | {4} |\n'.format('no link', *avg_prfs[:-1, 0], support0_f))
        f.write('| {0}    | {1:.2f}       | {2:.2f}   | {3:.2f}    | {4} |\n\n'.format('link', *avg_prfs[:-1, 1], support1_f))

