import importlib
import os
import sys
import numpy as np
from math import ceil
import torch
import torch.optim as optim
import torch.nn as nn
import visdom as visdom
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import src.models as models


# Returns filename for the saved model
def get_fname(config):
    duplicate_examples = config['duplicate_examples']
    repeat_examples = config['repeat_examples']
    agg_class = config['agg_class']
    hidden_dims_str = '_'.join([str(x) for x in config['hidden_dims']])
    num_samples = config['num_samples']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    fname = 'graphsage_agg_class_{}_hidden_dims_{}_num_samples_{}_batch_size_{}_epochs_{}_lr_{}_weight_decay_{}_' \
            'duplicate_{}_repeat_{}.pth'.format(
        agg_class, hidden_dims_str, num_samples, batch_size, epochs, lr,
        weight_decay, duplicate_examples, repeat_examples)

    return fname


# Returns a torch Dataset object
def get_dataset(args):
    dataset = None
    task, dataset_name, *dataset_args = args
    if task == 'link_prediction':
        class_attr = getattr(importlib.import_module('datasets.link_prediction'), dataset_name)
        dataset = class_attr(*dataset_args)

    return dataset


# Returns aggregator class
def get_agg_class(agg_class):
    return getattr(sys.modules[models.__name__], agg_class)


# Returns task name
def get_criterion(task):
    criterion = None
    if task == 'link_prediction':
        criterion = nn.BCEWithLogitsLoss()

    return criterion


# arguments - might change to parse from input
args = {
    "task": "link_prediction",

    "dataset": "IAContactsHypertext",
    "dataset_path": os.getcwd()[:len(os.getcwd()) - 3] + "data\\deep_learning\\djava_transformed",

    "mode": "train",
    "generate_neg_examples": True,

    "duplicate_examples": False,
    "repeat_examples": True,

    "self_loop": True,
    "normalize_adj": False,

    "cuda": True,
    # GAT or GraphSAGE
    "model": "GAT",
    "agg_class": "MaxPoolAggregator",
    "num_heads": [1, 1],
    "hidden_dims": [64],
    "dropout": 0,
    "num_samples": -1,

    "epochs": 9,
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 5e-4,
    "stats_per_batch": 3,
    "visdom": True,

    "load": False,
    "save": True
}
config = args
config['num_layers'] = len(config['hidden_dims']) + 1

if config['cuda'] and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
if config["model"] == "GAT":
    device = "cpu"
config['device'] = device


# Load database
dataset_args = (config['task'], config['dataset'], config['dataset_path'],
                config['generate_neg_examples'], 'train',
                config['duplicate_examples'], config['repeat_examples'],
                config['num_layers'], config['self_loop'],
                config['normalize_adj'])
dataset = get_dataset(dataset_args)

loader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                    shuffle=True, collate_fn=dataset.collate_wrapper)
input_dim, output_dim = dataset.get_dims()

# Load model
agg_class = get_agg_class(config['agg_class'])
model = None
if config["model"] == "GAT":
    model = models.GAT(input_dim, config['hidden_dims'],
                       output_dim, config['num_heads'],
                       config['dropout'], config['device'])
    model.apply(models.init_weights)
    model.to(config['device'])
elif config["model"] == "GraphSAGE":
    model = models.GraphSAGE(input_dim, config['hidden_dims'],
                         output_dim, config['dropout'],
                         agg_class, config['num_samples'],
                         config['device'])
    model.to(config['device'])
print(model)
if model is None:
    raise Exception("Invalid model type")

if not config['load']:

    # Compute ROC curve
    print('--------------------------------')
    print('Computing ROC-AUC score for the training dataset before training.')
    y_true, y_scores = [], []
    num_batches = int(ceil(len(dataset) / config['batch_size']))
    with torch.no_grad():
        for (idx, batch) in enumerate(loader):
            edges, features, node_layers, mappings, rows, labels = batch
            features, labels = features.to(device), labels.to(device)
            out = model(features, node_layers, mappings, rows)
            all_pairs = torch.mm(out, out.t())
            scores = all_pairs[edges.T]
            if config["model"] == "GAT":
                y_true.extend(labels.detach().numpy())
                y_scores.extend(scores.detach().numpy())
            if config["model"] == "GraphSAGE":
                y_true.extend(labels.detach().cpu().numpy())
                y_scores.extend(scores.detach().cpu().numpy())
            print('    Batch {} / {}'.format(idx + 1, num_batches))
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()
    area = roc_auc_score(y_true, y_scores)
    print('ROC-AUC score: {:.4f}'.format(area))
    print('--------------------------------')

    # TRAINING
    use_visdom = config['visdom']
    if use_visdom:
        vis = visdom.Visdom()
        loss_window = None
    criterion = get_criterion(config['task'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])
    epochs = config['epochs']
    stats_per_batch = config['stats_per_batch']
    num_batches = int(ceil(len(dataset) / config['batch_size']))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    scheduler  = None
    if config["model"] == "GAT":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
    elif config["model"] == "GraphSAGE":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 600], gamma=0.5)
    model.train()
    print('--------------------------------')
    print('Training.')
    for epoch in range(epochs):
        print('Epoch {} / {}'.format(epoch + 1, epochs))
        running_loss = 0.0
        for (idx, batch) in enumerate(loader):
            edges, features, node_layers, mappings, rows, labels = batch
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(features, node_layers, mappings, rows)
            all_pairs = torch.mm(out, out.t())
            scores = all_pairs[edges.T]
            loss = criterion(scores, labels.float())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
            if (idx + 1) % stats_per_batch == 0:
                running_loss /= stats_per_batch
                print('    Batch {} / {}: loss {:.4f}'.format(
                    idx + 1, num_batches, running_loss))
                if (torch.sum(labels.long() == 0).item() > 0) and (torch.sum(labels.long() == 1).item() > 0):
                    if config["model"] == "GAT":
                        area = roc_auc_score(labels.detach().numpy(), scores.detach().numpy())
                    elif config["model"] == "GraphSAGE":
                        area = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
                    print('    ROC-AUC score: {:.4f}'.format(area))
                running_loss = 0.0
                num_correct, num_examples = 0, 0
            if use_visdom:
                if loss_window is None:
                    loss_window = vis.line(
                        Y=[loss.item()],
                        X=[epoch * num_batches + idx],
                        opts=dict(xlabel='batch', ylabel='Loss', title='Training Loss', legend=['Loss']))
                else:
                    vis.line(
                        [loss.item()],
                        [epoch * num_batches + idx],
                        win=loss_window,
                        update='append')
            scheduler.step()
    if use_visdom:
        vis.close(win=loss_window)
    print('Finished training.')
    print('--------------------------------')

    # Save model if save mode is enabled
    if config['save']:
        print('--------------------------------')
        directory = os.path.join(os.path.dirname(os.getcwd()),
                                 'trained_models')
        if not os.path.exists(directory):
            os.makedirs(directory)
        fname = get_fname(config)
        path = os.path.join(directory, fname)
        print('Saving model at {}'.format(path))
        torch.save(model.state_dict(), path)
        print('Finished saving model.')
        print('--------------------------------')

    # AUC after training
    print('--------------------------------')
    print('Computing ROC-AUC score for the training dataset after training.')
    y_true, y_scores = [], []
    num_batches = int(ceil(len(dataset) / config['batch_size']))
    with torch.no_grad():
        for (idx, batch) in enumerate(loader):
            edges, features, node_layers, mappings, rows, labels = batch
            features, labels = features.to(device), labels.to(device)
            out = model(features, node_layers, mappings, rows)
            all_pairs = torch.mm(out, out.t())
            scores = all_pairs[edges.T]
            if config["model"] == "GAT":
                y_true.extend(labels.detach().numpy())
                y_scores.extend(scores.detach().numpy())
            elif config["model"] == "GraphSAGE":
                y_true.extend(labels.detach().cpu().numpy())
                y_scores.extend(scores.detach().cpu().numpy())
            print('    Batch {} / {}'.format(idx + 1, num_batches))
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()
    area = roc_auc_score(y_true, y_scores)
    print('ROC-AUC score: {:.4f}'.format(area))
    print('--------------------------------')

    # True positive and negative ratio plot
    tpr, fpr, thresholds = roc_curve(y_true, y_scores)
    tnr = 1 - fpr
    plt.plot(thresholds, tpr, label='tpr')
    plt.plot(thresholds, tnr, label='tnr')
    plt.xlabel('Threshold')
    plt.title('TPR / TNR vs Threshold')
    plt.legend()

    # Choose a threshold, generate classification report
    idx1 = np.where(tpr <= tnr)[0]
    idx2 = np.where(tpr >= tnr)[0]
    t = thresholds[idx1[-1]]
    total_correct, total_examples = 0, 0
    y_true, y_pred = [], []
    num_batches = int(ceil(len(dataset) / config['batch_size']))
    with torch.no_grad():
        for (idx, batch) in enumerate(loader):
            edges, features, node_layers, mappings, rows, labels = batch
            features, labels = features.to(device), labels.to(device)
            out = model(features, node_layers, mappings, rows)
            all_pairs = torch.mm(out, out.t())
            scores = all_pairs[edges.T]
            predictions = (scores >= t).long()
            if config["model"] == "GAT":
                y_true.extend(labels.detach().numpy())
                y_pred.extend(predictions.detach().numpy())
            elif config["model"] == "GraphSAGE":
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(predictions.detach().cpu().numpy())
            total_correct += torch.sum(predictions == labels.long()).item()
            total_examples += len(labels)
            print('    Batch {} / {}'.format(idx + 1, num_batches))
    print('Threshold: {:.4f}, accuracy: {:.4f}'.format(t, total_correct / total_examples))
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    report = classification_report(y_true, y_pred)
    print('Classification report\n', report)

    # Evaluate on validation set
    directory = os.path.join(os.path.dirname(os.getcwd()), 'trained_models')
    fname = get_fname(config)
    path = os.path.join(directory, fname)
    model.load_state_dict(torch.load(path))
dataset_args = (config['task'], config['dataset'], config['dataset_path'],
                config['generate_neg_examples'], 'val',
                config['duplicate_examples'], config['repeat_examples'],
                config['num_layers'], config['self_loop'],
                config['normalize_adj'])
dataset = get_dataset(dataset_args)
loader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                    shuffle=False, collate_fn=dataset.collate_wrapper)
criterion = get_criterion(config['task'])
stats_per_batch = config['stats_per_batch']
num_batches = int(ceil(len(dataset) / config['batch_size']))
model.eval()
print('--------------------------------')
print('Computing ROC-AUC score for the validation dataset after training.')
running_loss, total_loss = 0.0, 0.0
num_correct, num_examples = 0, 0
total_correct, total_examples = 0, 0
y_true, y_scores, y_pred = [], [], []
for (idx, batch) in enumerate(loader):
    edges, features, node_layers, mappings, rows, labels = batch
    features, labels = features.to(device), labels.to(device)
    out = model(features, node_layers, mappings, rows)
    all_pairs = torch.mm(out, out.t())
    scores = all_pairs[edges.T]
    loss = criterion(scores, labels.float())
    running_loss += loss.item()
    total_loss += loss.item()
    predictions = (scores >= t).long()
    num_correct += torch.sum(predictions == labels.long()).item()
    total_correct += torch.sum(predictions == labels.long()).item()
    num_examples += len(labels)
    total_examples += len(labels)
    if config["model"] == "GAT":
        y_true.extend(labels.detach().numpy())
        y_scores.extend(scores.detach().numpy())
        y_pred.extend(predictions.detach().numpy())
    elif config["model"] == "GraphSAGE":
        y_true.extend(labels.detach().cpu().numpy())
        y_scores.extend(scores.detach().cpu().numpy())
        y_pred.extend(predictions.detach().cpu().numpy())
    if (idx + 1) % stats_per_batch == 0:
        running_loss /= stats_per_batch
        accuracy = num_correct / num_examples
        print('    Batch {} / {}: loss {:.4f}, accuracy {:.4f}'.format(
            idx + 1, num_batches, running_loss, accuracy))
        if (torch.sum(labels.long() == 0).item() > 0) and (torch.sum(labels.long() == 1).item() > 0):
            area = None
            if config["model"] == "GAT":
                area = roc_auc_score(labels.detach().numpy(), scores.detach().numpy())
            elif config["model"] == "GraphSAGE":
                area = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
            print('    ROC-AUC score: {:.4f}'.format(area))
        running_loss = 0.0
        num_correct, num_examples = 0, 0
total_loss /= num_batches
total_accuracy = total_correct / total_examples
print('Loss {:.4f}, accuracy {:.4f}'.format(total_loss, total_accuracy))
y_true = np.array(y_true).flatten()
y_scores = np.array(y_scores).flatten()
y_pred = np.array(y_pred).flatten()
report = classification_report(y_true, y_pred)
area = roc_auc_score(y_true, y_scores)
print('ROC-AUC score: {:.4f}'.format(area))
print('Classification report\n', report)
print('Finished validating.')
print('--------------------------------')

# Evaluate on test set
if config['load']:
    directory = os.path.join(os.path.dirname(os.getcwd()), 'trained_models')
    fname = get_fname(config)
    path = os.path.join(directory, fname)
    model.load_state_dict(torch.load(path))
dataset_args = (config['task'], config['dataset'], config['dataset_path'],
                config['generate_neg_examples'], 'test',
                config['duplicate_examples'], config['repeat_examples'],
                config['num_layers'], config['self_loop'],
                config['normalize_adj'])
dataset = get_dataset(dataset_args)
loader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                    shuffle=False, collate_fn=dataset.collate_wrapper)
criterion = get_criterion(config['task'])
stats_per_batch = config['stats_per_batch']
num_batches = int(ceil(len(dataset) / config['batch_size']))
model.eval()
print('--------------------------------')
print('Computing ROC-AUC score for the test dataset after training.')
running_loss, total_loss = 0.0, 0.0
num_correct, num_examples = 0, 0
total_correct, total_examples = 0, 0
y_true, y_scores, y_pred = [], [], []
for (idx, batch) in enumerate(loader):
    edges, features, node_layers, mappings, rows, labels = batch
    features, labels = features.to(device), labels.to(device)
    out = model(features, node_layers, mappings, rows)
    all_pairs = torch.mm(out, out.t())
    scores = all_pairs[edges.T]
    loss = criterion(scores, labels.float())
    running_loss += loss.item()
    total_loss += loss.item()
    predictions = (scores >= t).long()
    num_correct += torch.sum(predictions == labels.long()).item()
    total_correct += torch.sum(predictions == labels.long()).item()
    num_examples += len(labels)
    total_examples += len(labels)
    if config["model"] == "GAT":
        y_true.extend(labels.detach().numpy())
        y_scores.extend(scores.detach().numpy())
        y_pred.extend(predictions.detach().numpy())
    elif config["model"] == "GraphSAGE":
        y_true.extend(labels.detach().cpu().numpy())
        y_scores.extend(scores.detach().cpu().numpy())
        y_pred.extend(predictions.detach().cpu().numpy())
    if (idx + 1) % stats_per_batch == 0:
        running_loss /= stats_per_batch
        accuracy = num_correct / num_examples
        print('    Batch {} / {}: loss {:.4f}, accuracy {:.4f}'.format(
            idx + 1, num_batches, running_loss, accuracy))
        if (torch.sum(labels.long() == 0).item() > 0) and (torch.sum(labels.long() == 1).item() > 0):
            if config["model"] == "GAT":
                area = roc_auc_score(labels.detach().numpy(), scores.detach().numpy())
            elif config["model"] == "GraphSAGE":
                area = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
            print('    ROC-AUC score: {:.4f}'.format(area))
        running_loss = 0.0
        num_correct, num_examples = 0, 0
total_loss /= num_batches
total_accuracy = total_correct / total_examples
print('Loss {:.4f}, accuracy {:.4f}'.format(total_loss, total_accuracy))
y_true = np.array(y_true).flatten()
y_scores = np.array(y_scores).flatten()
y_pred = np.array(y_pred).flatten()
report = classification_report(y_true, y_pred)
area = roc_auc_score(y_true, y_scores)
print('ROC-AUC score: {:.4f}'.format(area))
print('Classification report\n', report)
print('Finished testing.')
print('--------------------------------')
