import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

from z_flow.classifiers import Classifier


class SVM(Classifier):
    def __init__(self, type, interval, filter, method):
        super().__init__(type, interval, filter, method)
        self.feature_list = []
        self.label_list = []
        self.model = {}

    def train(self, cross_val=True, n_folds=5):
        feature_list = self.feature_list
        label_list = self.label_list
        if len(feature_list) < 5:
            print('only', len(feature_list), 'samples, please collect more!')
        else:
            X = np.array(feature_list)  # samples*features
            y = label_list
            clf = svm.SVC(probability=True)
            clf.fit(X, y)
            self.model['model'] = clf
            if cross_val is True:
                scores = cross_val_score(clf, X, y, cv=n_folds)
                self.model['cross validation accuracy'] = scores.mean()
            else:
                self.model['cross validation accuracy'] = None

    def predict(self, board_shim, board_id, event_timestamp):
        if 'model' in self.model.keys():
            sample_to_predict = np.array(self.collect_sample(board_shim, board_id, event_timestamp))
            clf = self.model['model']
            result = clf.predict([sample_to_predict])
            prob = clf.predict_proba([sample_to_predict])
            print('prediction result:', result, 'probability:', prob)
        else:
            print('no model trained yet, please train a model first!')
