""" Code for baseline implementation """
import os

import numpy as np
import pickle as pkl
import random
import time

from models import vrnn, birnn, cnn

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV as random_search
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score

PADDING_ID = 1016 # make the padding id as the number of group code
                  # maximum of group code index is 1015, start from 0

class SeqMethod(object):
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
        self.n_words = 1017
        self.n_classes = 2

    def load_data(self):
        with open(self.intmd_path + self.target + '.pos.pkl', 'rb') as f:
            X_pos, y_pos = pkl.load(f)
            f.close()

        with open(self.intmd_path + self.target + '.neg.pkl', 'rb') as f:
            X_neg, y_neg = pkl.load(f)
            f.close()

        print ("The number of positive samles is: ", len(y_pos))
        print ("The number of negative samles is: ", len(y_neg))

        # aggregate (and normalize) the data
        n_codes_pos = []
        n_codes_neg = []
        seq_len_pos = []
        seq_len_neg = []
        max_indice = []
        for s, array in X_pos.items():
             self.X_pos.append(array) # X_pos_mat[s] size: seq_len x n_words
             self.y_pos.append(y_pos[s])
             timesteps = array.shape[0]
             seq_len_pos.append(timesteps)
             # compute code size for postives
             # count_code = np.zeros(timesteps)
             # for i in range(timesteps):
             #     count_code[i] = 0
             #     for j in range(self.n_words-1):
             #         if X_pos[s][i][j] != PADDING_ID: count_code[i] += 1
             # n_codes_pos.append(np.max(count_code))
        for s, array in X_neg.items():
             self.X_neg.append(array)
             self.y_neg.append(y_neg[s])
             timesteps = array.shape[0]
             seq_len_neg.append(timesteps)
             max_indice.append(np.max(array))
             # compute code size for negatives
             # count_code = np.zeros(timesteps)
             # for i in range(timesteps):
             #     count_code[i] = 0
             #     for j in range(self.n_words-1):
             #         if X_neg[s][i][j] != PADDING_ID: count_code[i] += 1
             # n_codes_neg.append(np.max(count_code))
        self.timesteps = int(max(np.mean(seq_len_pos), np.mean(seq_len_neg)))
        print ("The selected timesteps is: ", self.timesteps)

        # self.code_size = int(max(np.max(n_codes_pos), np.max(n_codes_neg)))
        # save code_size
        # f = open(self.intmd_path + self.target + '.code.size.pkl', 'wb')
        # pkl.dump(self.code_size, f, protocol=2)
        # f.close()
        # open code_size
        f = open(self.intmd_path + self.target + '.code.size.pkl', 'rb')
        self.code_size = pkl.load(f)
        f.close()
        print ("The code_size is: ", self.code_size)
        return (self.X_pos, self.X_neg), (self.y_pos, self.y_neg)

    def get_fixed_timesteps(self):
        '''delete the first several timesteps according to the selected number'''
        # postives:
        for i in range(len(self.X_pos)):
            timesteps = self.X_pos[i].shape[0]
            if timesteps > self.timesteps:
                self.X_pos[i] = self.X_pos[i][timesteps-self.timesteps:, :]

        # negatives:
        for i in range(len(self.X_neg)):
            timesteps = self.X_neg[i].shape[0]
            if timesteps > self.timesteps:
                self.X_neg[i] = self.X_neg[i][timesteps-self.timesteps:, :]
        return (self.X_pos, self.X_neg)

    def get_fixed_codesize(self):
        '''delete the -1 values according to the code size'''
        # postives:
        for i in range(len(self.X_pos)):
            code_size = self.X_pos[i].shape[1]
            if code_size > self.code_size:
                self.X_pos[i] = self.X_pos[i][:, :self.code_size]
        # negatives:
        for i in range(len(self.X_neg)):
            code_size = self.X_neg[i].shape[1]
            if code_size > self.code_size:
                self.X_neg[i] = self.X_neg[i][:, :self.code_size]
        return (self.X_pos, self.X_neg)

    def get_feed_records(self, X):
        '''generate ehrs as a 3d tensor that can be used to feed networks'''
        n_samples = len(X)
        X_new = np.zeros([n_samples, self.timesteps, self.code_size], dtype="int32") + PADDING_ID
        for i in range(n_samples):
            timesteps = X[i].shape[0]
            X_new[i, self.timesteps-timesteps:, :] = X[i]
        return X_new

    def get_classifiers(self, X, y):
        '''split by StratifiedKFold, then use lr, svm, rf, gbdt and mlp classifiers.
        lr, svm, mlp need normalization
        '''
        X_pos, X_neg = X
        y_pos, y_neg = y

        X_pos = self.get_feed_records(X_pos)
        X_neg = self.get_feed_records(X_neg)
        X, y = np.concatenate((X_pos, X_neg), axis=0), np.concatenate((y_pos, y_neg), axis=0)

        #########################
        p = np.random.permutation(len(X))
        X,y = X[p],y[p]
        n_fold = 5
        skf = StratifiedKFold(n_splits = n_fold, random_state = 99991)
        scaler = StandardScaler()
        # OPTION: choose one of the neural nets
        model_choices = {"RNN":vrnn, "BiRNN":birnn, "CNN":cnn}
        ifold = 0
        Res = dict()
        for method in model_choices:
            Res[method] = {'aucroc': [], 'spec': [], 'sen': [], 'aucprc': [], 'avepre': [], 'f1score': []}

        for train_index, test_index in skf.split(X,y):
            ifold+=1
            print ("----------The %d-th fold-----------" %ifold)
            X_tr, X_te = X[train_index], X[test_index]
            y_tr, y_te = y[train_index], y[test_index]

            for k, m in model_choices.items():
                print ("The current model for optimizing is: " + k)
                #train
                dir_name = str(m)
                # init: feature_dim, num_classes, code_size
                model = m(self.n_words, self.n_classes, self.timesteps, self.code_size, dir_name)
                fit_auc, fit_accuracy, fit_losses = model.fit(X_tr, y_tr, X_te, y_te)
                string, auc, accuracy, loss, yhat = model.evaluate(X_te, y_te)

                #eval: aucroc, aucprc
                aucroc = roc_auc_score(y_te, yhat)
                avepre = average_precision_score(y_te, yhat)
                tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
                f1score = f1_score(y_te, yhat, 'micro')

                # true negative, false positive, false negative, true positive
                spec = tn / (tn+fp)
                sen = tp / (tp+fn)

                Res[k]['aucroc'].append(aucroc)
                Res[k]['spec'].append(spec)
                Res[k]['sen'].append(sen)
                Res[k]['avepre'].append(avepre)
                Res[k]['f1score'].append(f1score)

        # show results
        for method in model_choices:
            print ("----------")
            print (method + ":")
            print ('aucroc mean: ', np.mean(np.array(Res[method]['aucroc'])))
            print ('aucroc std: ', np.std(np.array(Res[method]['aucroc'])))
            print ('spec mean: ', np.mean(np.array(Res[method]['spec'])))
            print ('spec std: ', np.std(np.array(Res[method]['spec'])))
            print ('sen mean: ', np.mean(np.array(Res[method]['sen'])))
            print ('sen std: ', np.std(np.array(Res[method]['sen'])))
            print ('avepre mean: ', np.mean(np.array(Res[method]['avepre'])))
            print ('avepre std: ', np.std(np.array(Res[method]['avepre'])))
            print ('f1score mean: ', np.mean(np.array(Res[method]['f1score'])))
            print ('f1score std: ', np.std(np.array(Res[method]['f1score'])))


def main():
    target = "AD"
    sm = SeqMethod(target)
    X, y = sm.load_data()
    X = sm.get_fixed_timesteps()
    X = sm.get_fixed_codesize()
    sm.get_classifiers(X, y)

if __name__ == "__main__":
    main()
