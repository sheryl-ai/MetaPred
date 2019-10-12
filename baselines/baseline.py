""" Code for baseline implementation """
import os

import numpy as np
import pickle as pkl
import random
import time

from classifiers import *
from mlp import MLP

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV as random_search
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, auc, roc_curve, f1_score



class Baseline(object):
    """
    Classifiers: lr, svm, rf, gbdt, mlp.
    """
    def __init__(self, target, config={}):
        """
        Args:
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.X_pos, self.y_pos = [], []
        self.X_neg, self.y_neg = [], []
        self.intmd_path = 'intermediate/'
        self.target = target

    def load_data(self):
        with open(self.intmd_path + self.target + '.pos.mat.pkl', 'rb') as f:
            X_pos_mat, y_pos_mat = pkl.load(f)
            f.close()

        with open(self.intmd_path + self.target + '.neg.mat.pkl', 'rb') as f:
            X_neg_mat, y_neg_mat = pkl.load(f)
            f.close()

        print ("The number of positive samles is: ", len(y_pos_mat))
        print ("The number of negative samles is: ", len(y_neg_mat))

        # aggregate (and normalize) the data
        for s, array in X_pos_mat.items():
             self.X_pos.append(np.sum(X_pos_mat[s], axis=0))
             self.y_pos.append(y_pos_mat[s])
        for s, array in X_neg_mat.items():
             self.X_neg.append(np.sum(X_neg_mat[s], axis=0))
             self.y_neg.append(y_neg_mat[s])

        return (self.X_pos, self.X_neg), (self.y_pos, self.y_neg)

    def get_classifiers(self, X, y):
        '''split by StratifiedKFold, then use lr, svm, rf, gbdt and mlp classifiers.
        lr, svm, mlp need normalization
        '''
        X_pos, X_neg = X
        y_pos, y_neg = y
        X, y = np.concatenate((X_pos, X_neg), axis=0), np.concatenate((y_pos, y_neg), axis=0)
        p = np.random.permutation(len(X))
        X,y = X[p],y[p]

        n_fold = 5
        skf = StratifiedKFold(n_splits = n_fold, random_state = 99991)
        scaler = StandardScaler()
        # OPTION: choose one of the classifiers
        models = {"LR":lr, "KNN":knn, "SVM":svm, "RF":rf, "XGB":xgb, "MLP":MLP}
        ifold = 0
        results = dict()
        Res = {'aucroc': [], 'spec': [], 'sen': [], 'aucprc': [], 'avepre': [], 'f1score': []}
        for train_index, test_index in skf.split(X,y):
            ifold+=1
            print ("----------The %d-th fold-----------" %ifold)
            results[ifold] = dict()

            X_tr, X_te = X[train_index], X[test_index]
            y_tr, y_te = y[train_index], y[test_index]

            for k, m in models.items():
                print ("The current model for optimizing is: " + k)
                #train
                if k == "MLP":
                    # init: feature_dim, num_classes
                    mlp = m(X_tr.shape[1], 2)
                    fit_auc, fit_accuracy, fit_losses = mlp.fit(X_tr, y_tr, X_te, y_te)
                    string, auc, accuracy, loss, yhat = mlp.evaluate(X_te, y_te)
                    yhat = np.array(yhat, dtype="float32")
                else:
                    m = m.fit(X_tr, y_tr)
                    yhat = m.predict(X_te)
                #eval: aucroc, aucprc
                aucroc = roc_auc_score(y_te, yhat)
                avepre = average_precision_score(y_te, yhat)
                tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
                f1score = f1_score(y_te, yhat, 'micro')

                # true negative, false positive, false negative, true positive
                spec = tn / (tn+fp)
                sen = tp / (tp+fn)
                models[k] = m
                Res['aucroc'].append(aucroc)
                Res['spec'].append(spec)
                Res['sen'].append(sen)
                Res['aucprc'].append(aucprc)
                Res['avepre'].append(avepre)
                Res['f1score'].append(f1score)

        print ('aucroc mean: ', np.mean(np.array(Res['aucroc'])))
        print ('aucroc std: ', np.std(np.array(Res['aucroc'])))
        print ('spec mean: ', np.mean(np.array(Res['spec'])))
        print ('spec std: ', np.std(np.array(Res['spec'])))
        print ('sen mean: ', np.mean(np.array(Res['sen'])))
        print ('sen std: ', np.std(np.array(Res['sen'])))
        print ('avepre mean: ', np.mean(np.array(Res['avepre'])))
        print ('avepre std: ', np.std(np.array(Res['avepre'])))
        print ('f1score mean: ', np.mean(np.array(Res['f1score'])))
        print ('f1score std: ', np.std(np.array(Res['f1score'])))

#### Hyperparams Search ####
#######################
def classic_rsearch(x,y):
    from scipy.stats import uniform as sp_rand
    from scipy.stats import randint as sp_randint
    lr1 = LR(warm_start = True, penalty = 'l1', verbose = 100, max_iter = 5000)
    lr2 = LR(warm_start = True, penalty = 'l2', verbose = 100, max_iter = 5000)
    svm = SVM(verbose = True, probability = False, max_iter= 5000)
    rf = RF(warm_start = True, verbose = 100)

    #random search params
    lr_params = {'C': sp_rand(1, 1e5)}
    rf_params = {'criterion': ['gini', 'entropy'], 'n_estimators': sp_randint(10, 200), 'max_features': ['auto', 'sqrt', 'log2', None]}
    mlp_params = {'hidden_layer_sizes':[(64, 64), (128, 128), (256, 256), (512, 512)], 'alpha': sp_rand(1e-6, 1e-2)}
    svm_params = {'kernel': ['rbf', 'poly'], 'C':sp_rand (1, 1e5), 'gamma': sp_rand(1e-5, 1)}

    results = {}; models = []
    lst = [lr1, lr2, svm, rf]
    names = ['LR','SVM','RF']
    params = [lr_params, lr_params, svm_params, rf_params]
    for idx in range(len(lst)):
        n_iter_search = 60
        start = time.time()
        rsearch = random_search(estimator = lst[idx], param_distributions = params[idx], n_iter=n_iter_search,
                                scoring='roc_auc', fit_params=None, n_jobs=1,
                                iid=True, refit=True, cv=5, verbose=0, random_state=8)
        rsearch.fit(x, y)
        models.append(rsearch)
        results[names[idx]] = rsearch.cv_results_
        print (names[idx]+" results complete.")
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time.time() - start), n_iter_search))
    return (data, models)


def main():
    target = "AD"
    bl = Baseline(target)
    X, y = bl.load_data()
    bl.get_classifiers(X, y)

if __name__ == "__main__":
    main()
