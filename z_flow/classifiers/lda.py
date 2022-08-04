import numpy as np

from z_flow.classifiers import Classifier


class LDA(Classifier):
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
            self.model['parameter-w,b'] = self.train_LDA(feature_list, label_list)
            if cross_val is True:
                n_sample = len(feature_list)
                n_in_fold = np.floor(n_sample/n_folds).astype(int)
                n_sample = n_in_fold * n_folds
                perm = np.random.permutation(n_sample)
                acc_test = []

                # cross validation
                for i_fold in range(n_folds):
                    # train for each fold
                    idx_test = perm[i_fold * n_in_fold:(i_fold + 1) * n_in_fold]
                    idx_train = np.setdiff1d(range(n_sample), idx_test)
                    feature_list_train = np.array(feature_list)[idx_train].tolist()
                    label_list_train = np.array(label_list)[idx_train].tolist()
                    label_list_test = np.array(label_list)[idx_test].tolist()
                    w, b = self.train_LDA(feature_list_train, label_list_train)

                    # predict and calculate accuracy
                    prediction = w.T.dot(np.array(feature_list).T) - b
                    prediction_test = prediction[idx_test]
                    accuracy_test = (sum(prediction_test[np.array(label_list_test) == 0] < 0) + sum(
                        prediction_test[np.array(label_list_test) == 1] >= 0)) / len(prediction_test)
                    acc_test.append(accuracy_test)
                self.model['cross validation accuracy'] = np.mean(acc_test)
            else:
                self.model['cross validation accuracy'] = None

    def train_LDA(self, feature_list, label_list):
        X = np.array(feature_list).T  # features*samples
        y = np.array(label_list)
        mu1 = np.mean(X[:, y == 1], axis=1)
        mu0 = np.mean(X[:, y == 0], axis=1)
        # center features to estimate covariance
        Xpool = np.concatenate((X[:, y == 1] - mu1[:, np.newaxis], X[:, y == 0] - mu0[:, np.newaxis]), axis=1)
        C = np.cov(Xpool)
        w = np.linalg.pinv(C).dot(mu1 - mu0)
        b = w.T.dot((mu1 + mu0) / 2)
        return w, b

    def predict(self, board_shim, board_id, event_timestamp):
        if 'parameter-w,b' in self.model.keys():
            sample_to_predict = np.array(self.collect_sample(board_shim, board_id, event_timestamp)).T
            w, b = self.model['parameter-w,b']
            result = w.T.dot(sample_to_predict) - b
            print('prediction result:', result)
        else:
            print('no model trained yet, please train a model first!')
