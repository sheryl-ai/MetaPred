""" Code for data loader """
import numpy as np
import os, sys, copy
import random
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from tensorflow.python.platform import flags

import tqdm
import pickle as pkl

FLAGS = flags.FLAGS

PADDING_ID = 1016 # make the padding id as the number of group code
                  # maximum of group code index is 1015, start from 0
N_WORDS = 1017
TIMESTEPS = 21 # choice by statistics

TASKS = ["AD", "PD", "DM", "AM", "MCI"]

class DataLoader(object):
    '''
    Data Loader capable of generating batches of ohsu data.
    '''
    def __init__(self, source, target, true_target, n_tasks, n_samples_per_task, meta_batch_size):
        """
        Args:
            source:             source tasks
            target:             simulated target task(s)
            true_target:        true target task (to test)
            n_tasks:            number of tasks including both source and simulated target tasks
            n_samples_per_task: number samples to generate per task in one batch
            meta_batch_size:    size of meta batch size (e.g. number of functions)
        """
        ### load data: training
        self.intmd_path = 'intermediate/'
        self.source = source
        self.target = target
        self.timesteps = TIMESTEPS
        self.code_size = 0
        # self.code_size = N_WORDS-1 # set the code_size as the number of all the possible codes
        #                            # in order to use in pretrain
        self.task_code_size = dict() # maintain a dictionary for icd codes, disease : code list
        print ("The selected timesteps is: ", self.timesteps)

        self.data_to_show = dict()
        self.label_to_show = dict()
        self.ratio_t = 0.8
        self.pat_reduce = False
        self.code_set = set()
        self.data_s, self.data_t, self.label_s, self.label_t = self.load_data()

        ## load data: validate & test
        self.true_target = true_target
        if FLAGS.method == "mlp":
            data_tt, label_tt = self.load_data_vector(self.true_target[0]) # only 1 true target, index is 0
        elif FLAGS.method == "rnn" or FLAGS.method == "cnn":
            data_tt, label_tt = self.load_data_matrix(self.true_target[0])
            # compute code_size
            self.code_size = max([cz for cz in self.task_code_size.values()])
            print ("The code_size is: ", self.code_size)
            # make data the same size matrices
            data_tt, label_tt = self.get_data_prepared(data_tt, label_tt)

            for i in range(len(self.source)):
                self.data_s[i], self.label_s[i] = self.get_data_prepared(self.data_s[i], self.label_s[i])

            for i in range(len(self.target)):
                self.data_t[i], self.label_t[i] = self.get_data_prepared(self.data_t[i], self.label_t[i])

        # cross validation for true target
        self.n_fold = 5
        self.get_cross_val(data_tt, label_tt, n_fold=self.n_fold)

        ### set model params
        self.meta_batch_size = meta_batch_size
        self.n_samples_per_task = n_samples_per_task # in one meta batch
        self.n_tasks = n_tasks
        self.n_words = N_WORDS

        ## generate finetune data
        self.tt_sample, self.tt_label = dict(), dict()
        self.tt_sample_val, self.tt_label_val = dict(), dict()
        for ifold in range(self.n_fold): # generate n-fold cv data for finetuning
            self.tt_sample[ifold], self.tt_label[ifold] = self.generate_finetune_data(is_training=True, ifold=ifold)
            self.tt_sample_val[ifold], self.tt_label_val[ifold] = self.generate_finetune_data(is_training=False, ifold=ifold)

        self.episode = self.generate_meta_idx_batches(is_training=True)
        self.episode_val = dict()
        for ifold in range(self.n_fold): # true target validation
            self.episode_val[ifold] = self.generate_meta_idx_batches(is_training=False, ifold=ifold)

    def get_cross_val(self, X, y, n_fold=5):
        '''split the true target into train (might be useful in finetunning) and test (for evaluation)'''
        self.data_tt_tr, self.data_tt_val = dict(), dict()
        self.label_tt_tr, self.label_tt_val = dict(), dict()
        skf = StratifiedKFold(n_splits = n_fold, random_state = 99991)
        ifold = 0
        print ("split the true target ...")
        for train_index, test_index in skf.split(X, y):
            self.data_tt_tr[ifold], self.data_tt_val[ifold] = X[train_index], X[test_index]
            self.label_tt_tr[ifold], self.label_tt_val[ifold] = y[train_index], y[test_index]
            ifold+=1

    def load_data_matrix(self, task):
        '''load data sequential vectors for cnn or rnn. One matrix per sample'''
        X_pos, y_pos = [], []
        X_neg, y_neg = [], []
        with open(self.intmd_path + task + '.pos.pkl', 'rb') as f:
            X_pos_mat, y_pos_mat = pkl.load(f)
            f.close()

        with open(self.intmd_path + task + '.neg.pkl', 'rb') as f:
            X_neg_mat, y_neg_mat = pkl.load(f)
            f.close()

        print ("The number of positive samles in task %s is: " %task, len(y_pos_mat))
        print ("The number of negative samles in task %s is: " %task, len(y_neg_mat))

        for s, array in X_pos_mat.items():
             X_pos.append(array) # X_pos_mat[s] size: seq_len x n_words
             y_pos.append(y_pos_mat[s])

        for s, array in X_neg_mat.items():
             X_neg.append(array)
             y_neg.append(y_neg_mat[s])
        return (X_pos, X_neg), (y_pos, y_neg)

    def get_fixed_timesteps(self, X_pos, X_neg):
        '''delete the first several timesteps according to the selected number'''
        # postives:
        for i in range(len(X_pos)):
            timesteps = X_pos[i].shape[0]
            if timesteps > self.timesteps:
                X_pos[i] = X_pos[i][timesteps-self.timesteps:, :]
        # negatives:
        for i in range(len(X_neg)):
            timesteps = X_neg[i].shape[0]
            if timesteps > self.timesteps:
                X_neg[i] = X_neg[i][timesteps-self.timesteps:, :]
        return (X_pos, X_neg)

    def get_fixed_codesize(self, X_pos, X_neg):
        '''delete the -1 values according to the code size'''
        # postives:
        for i in range(len(X_pos)):
            code_size = X_pos[i].shape[1]
            if code_size > self.code_size:
                X_pos[i] = X_pos[i][:, :self.code_size]
        # negatives:
        for i in range(len(X_neg)):
            code_size = X_neg[i].shape[1]
            if code_size > self.code_size:
                X_neg[i] = X_neg[i][:, :self.code_size]
        return (X_pos, X_neg)

    def get_feed_records(self, X):
        '''generate ehrs as a 3d tensor that can be used to feed networks'''
        n_samples = len(X)
        X_new = np.zeros([n_samples, self.timesteps, self.code_size], dtype="int32") + PADDING_ID
        for i in range(n_samples):
            timesteps = X[i].shape[0]
            X_new[i, self.timesteps-timesteps:, :] = X[i]
        return X_new

    def get_data_prepared(self, data, label):
        X_pos, X_neg = data
        y_pos, y_neg = label

        X_pos, X_neg = self.get_fixed_timesteps(X_pos, X_neg)
        X_pos, X_neg = self.get_fixed_codesize(X_pos, X_neg)
        X_pos = self.get_feed_records(X_pos)
        X_neg = self.get_feed_records(X_neg)
        # concatenate pos and neg
        data, label = np.concatenate((X_pos, X_neg), axis=0), np.concatenate((y_pos, y_neg), axis=0)
        return data, label

    def load_data(self):
        '''load data vectors or matrices for samples with labels'''
        data_s, label_s = dict(), dict()
        data_t, label_t = dict(), dict()

        self.dim_input = [TIMESTEPS, N_WORDS]
        for i in range(len(self.source)):
            data_s[i], label_s[i] = self.load_data_matrix(self.source[i])

        for i in range(len(self.target)):
            data_t[i], label_t[i] = self.load_data_matrix(self.target[i])
        return data_s, data_t, label_s, label_t

    def generate_finetune_data(self, is_training=True, ifold=0):
        ''' get finetuning samples and labels'''
        try:
            if is_training:
                sample = self.data_tt_tr[ifold]
                label = self.label_tt_tr[ifold]
            else:
                sample = self.data_tt_val[ifold]
                label = self.label_tt_val[ifold]
        except:
            print ("Error: split training and validate first!")
        return sample, label

    def generate_meta_batches(self, is_training=True, ifold=0):
        ''' get samples and the corresponding labels with episode for batching'''
        if is_training: # training
            prefix = "metatrain"
            data_s = self.data_s
            data_t = self.data_t
            label_s = self.label_s
            label_t = self.label_t
            self.n_total_batches = FLAGS.n_total_batches
        else: # test & eval, say, true target task is used here
            try:
                prefix = "metaval" + str(ifold)
                data_s = self.data_s
                label_s = self.label_s
                data_t = self.data_tt_val[ifold]
                label_t = self.label_tt_val[ifold]
                self.n_total_batches = int(len(label_t)/self.n_samples_per_task)
            except:
                print ("Error: split training and validate first!")
        # check if the meta batch file dumped
        if os.path.isfile(self.intmd_path + "meta.batch." + prefix + ".pkl"):
            print ('meta batch file exits')
            with open(self.intmd_path + "meta.batch." + prefix + ".pkl", 'rb') as f:
                sample, label = pkl.load(f)
                f.close()
        else:
            # generate episode
            sample, label = [], []
            s_dict, t_dict = dict(), dict()
            for i in range(len(self.source)):
                s_dict[i] = range(len(self.label_s[i]))
            for i in range(len(self.target)):
                t_dict[i] = range(len(self.label_t[i]))
            batch_count = 0
            for _ in tqdm.tqdm(range(self.n_total_batches), 'generating meta batches'): # progress bar
                # i.e., sample 16 patients from selected tasks
                # len of spl and lbl: 4 * 16
                spl, lbl = [], [] # samples and labels in one episode
                for i in range(len(self.source)): # fetch from source tasks olderly
                    ### do not keep pos/neg ratio
                    s_idx = random.sample(s_dict[i], self.n_samples_per_task)
                    spl.extend(data_s[i][s_idx])
                    lbl.extend(label_s[i][s_idx])
                ### do not keep pos/neg ratio
                if is_training:
                    t_idx = random.sample(t_dict[0], self.n_samples_per_task)
                    spl.extend(data_t[0][t_idx])
                    lbl.extend(label_t[0][t_idx])
                else:
                    spl.extend(data_t[batch_count*self.n_samples_per_task:(batch_count+1)*self.n_samples_per_task])
                    lbl.extend(label_t[batch_count*self.n_samples_per_task:(batch_count+1)*self.n_samples_per_task])
                batch_count += 1
                # add meta_batch
                sample.append(spl)
                label.append(lbl)

        print ("batch counts: ", batch_count)
        sample = np.array(sample, dtype="float32")
        label = np.array(label, dtype="float32")
        return sample, label

    def generate_meta_idx_batches(self, is_training=True, ifold=0):
        ''' get samples and the corresponding labels with episode for batching'''
        if is_training: # training
            prefix = "metatrain"
            data_s = self.data_s
            data_t = self.data_t
            label_s = self.label_s
            label_t = self.label_t
            self.n_total_batches = FLAGS.n_total_batches
        else: # test & eval, say, true target task is used here
            try:
                prefix = "metaval" + str(ifold)
                data_s = self.data_s
                label_s = self.label_s
                data_t = self.data_tt_val[ifold]
                label_t = self.label_tt_val[ifold]
                self.n_total_batches = int(len(label_t)/self.n_samples_per_task)
                print (data_t.shape)
                print (label_t.shape)
                print (len(label_t))
            except:
                print ("Error: split training and validate first!")

        # generate episode
        episode = []
        s_dict, t_dict = dict(), dict()
        for i in range(len(self.source)):
            s_dict[i] = range(len(self.label_s[i]))
        for i in range(len(self.target)):
            t_dict[i] = range(len(self.label_t[i]))
        batch_count = 0
        for _ in tqdm.tqdm(range(self.n_total_batches), 'generating meta batches'): # progress bar
            # i.e., sample 16 patients from selected tasks
            # len of spl and lbl: 4 * 16
            idx = [] # index in one episode
            for i in range(len(self.source)): # fetch from source tasks olderly
                ### do not keep pos/neg ratio
                s_idx = random.sample(s_dict[i], self.n_samples_per_task)
                idx.extend(s_idx)
            ### do not keep pos/neg ratio
            if is_training:
                t_idx = random.sample(t_dict[0], self.n_samples_per_task)
                idx.extend(t_idx)
            else:
                t_idx = range(batch_count*self.n_samples_per_task, (batch_count+1)*self.n_samples_per_task)
                idx.extend(t_idx)
            batch_count += 1
            # add meta_batch
            episode.append(idx)

        print ("batch counts: ", batch_count)
        return episode
