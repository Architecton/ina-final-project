import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Gradient boosting classifier implementation

    Args:
        params (dict): Model parameters. If None, use default pre-set values.
        objective (str): Prediction objecitve. Currently, only 'binary:*' and 'multi:softprob'
        have been tested.
        n_rounds (int): Number of training rounds.
        num_class (int): Number of different classes if performing multi-class classification.

    Attributes:
        params (dict): Model parameters. If None, use default pre-set values.
        objective (str): Prediction objecitve. Currently, only 'binary:*' and 'multi:softprob'
        have been tested.
        n_rounds (int): Number of training rounds.
        num_class (int): Number of different classes if performing multi-class classification.
    """

    def __init__(self, params=None, objective='binary:logistic', n_rounds=1000, num_class=-1):

        # Set parameters.
        if params is None:
            self.params = {
                    'max_depth' : 4,
                    'eta' : 0.3,
                    'silent' : 1,
                    'objective': objective,
                    }
        else:
            self.params = params
        
        # Set prediction objective.
        self.objective = objective

        # Set number of classes if doint multi-label classification.
        if self.objective[:5] == 'multi':
            self.params['num_class'] = num_class

        # Set number of rounds.
        self.n_rounds = n_rounds


    def fit(self, X, y):
        """
        Fit classifier to training data.

        Args:
            X (numpy.ndarray): Training data samples
            y (numpy.ndarray): Training data labels

        Returns:
            (obj): Reference to self
        """
        
        # Split training data into training and validation sets.
        data_train, data_val, target_train, target_val = train_test_split(X, y, test_size=0.2)

        # Define train and validation sets in required format.
        dtrain = xgb.DMatrix(data_train, target_train)
        dval = xgb.DMatrix(data_val, target_val)
        watchlist = [(dval, 'eval'), (dtrain, 'train')]

        # Train model.
        gbm = xgb.train(self.params,
                        dtrain,
                        num_boost_round = self.n_rounds,
                        evals = watchlist,
                        verbose_eval = True
                        )
        self._gbm = gbm
        
        # Return reference to self.
        return self

  
    def predict(self, X, y=None):
        """
        Predict labels of new data.
        
        Args:
            X (numpy.ndarray): Data for which to predict classes

        Returns:
            (numpy.ndarray): Predicted labels
        """

        # Return labels with highest probability.
        if self.objective[:6] == 'binary':
            return np.where(self._gbm.predict(xgb.DMatrix(X)) > 0.5, 1, 0)
        elif self.objective[:5] == 'multi':
            return np.argmax(self._gbm.predict(xgb.DMatrix(X)), axis=1)
        else:
            raise(NotImplementedError('Other objectives not yet tested'))
   

    def predict_proba(self, X, y=None):
        """
        Predict labels of new data (return probabilities).
        
        Args:
            X (numpy.ndarray): Data for which to predict classes

        Returns:
            (numpy.ndarray): Predicted label probabilities
        """
       
        # Return probabilities of labels.
        if self.objective[:6] == 'binary':
            probs = self._gbm.predict(xgb.DMatrix(X))
            return np.vstack((1-probs, probs)).T
        elif self.objective[:5] == 'multi':
            return self._gbm.predict(xgb.DMatrix(X))
        else:
            raise(NotImplementedError('Other objectives not yet tested'))


    def score(self, X, y, sample_weight=None):
        """
        Score predictions on training data.
        
        Args:
            X (numpy.ndarray): Test data
            y (numpy.ndarray): True labels
            sample_weight (numpy.ndarray): Sample weights for scoring predictions

        Returns:
            (float): Accuracy scores of predictions
        """
        
        # Score predictions.
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    
    def score_features(self, f_to_name):
        """
        Score feature importances.

        Args:
            f_to_name (dict): Dictionary mapping feature enumerations such as 'f0', 'f1', ...
            to feature names.

        Returns:
            (dict): Dictionary mapping feature names as defined in f_to_name parameter to
            their estimated importances.
        """

        f_scores = self._gbm.get_fscore()
        sum_f_scores = sum(f_scores.values())
        return {f_to_name[key] : val/sum_f_scores for key, val in f_scores.items()}

