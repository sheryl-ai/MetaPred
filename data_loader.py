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

        # self.code_set = set(self.code_set)
        # print (len(self.code_set))
        # with open("useful.code.pkl", "wb") as f:
        #     pkl.dump(self.code_set, f, protocol=2)
        #     f.close()

        # print (self.data_to_show)
        # print (self.label_to_show)
        # cross validation for true target
        self.n_fold = 5
        self.get_cross_val(data_tt, label_tt, n_fold=self.n_fold)

        ### set model params
        self.meta_batch_size = meta_batch_size
        self.n_samples_per_task = n_samples_per_task # in one meta batch
        # self.n_pos_per_task = int(n_samples_per_task * self.pos_ratio) # in one meta batch, keep the ratio according to target task (pos/(pos+neg))
        # self.n_neg_per_task = self.n_samples_per_task - self.n_pos_per_task
        self.n_tasks = n_tasks
        self.n_words = N_WORDS

        ## generate pretrain or finetune data
        self.tt_sample, self.tt_label = dict(), dict()
        self.tt_sample_val, self.tt_label_val = dict(), dict()
        for ifold in range(self.n_fold): # generate n-fold cv data for pretraining
            self.tt_sample[ifold], self.tt_label[ifold] = self.generate_pretrain_data(is_training=True, ifold=ifold)
            self.tt_sample_val[ifold], self.tt_label_val[ifold] = self.generate_pretrain_data(is_training=False, ifold=ifold)

        self.episode = self.generate_meta_idx_batches(is_training=True)
        self.episode_val = dict()
        for ifold in range(self.n_fold): # true target validation is consistent with pretraining
            self.episode_val[ifold] = self.generate_meta_idx_batches(is_training=False, ifold=ifold)

        ## generate meta batches, for training and validation/test
        # self.sample, self.label = self.generate_meta_batches(is_training=True)
        # self.sample_val, self.label_val = dict(), dict()
        # for ifold in range(self.n_fold): # true target validation is consistent with pretraining
        #     self.sample_val[ifold], self.label_val[ifold] = self.generate_meta_batches(is_training=False, ifold=ifold)

        # in order to get sample representations for some specific tasks
        if FLAGS.rept:
            episode_name= dict()
            task_list = copy.deepcopy(TASKS)
            # task_list.remove(FLAGS.source)
            # task_list.remove(FLAGS.target)
            self.task_list_rept = task_list
            for task in task_list:
                episode_name[task] = []
            self.load_data_to_show(task_list)
            self.episode_rep = self.generate_meta_idx_batches(is_training=False, is_represent=True, episode_name=episode_name)


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

    def load_data_to_show(self, task_list):
        # only consider mlp now
        # common_pat = dict()
        for task in task_list:
            # with open(self.intmd_path + task + '.common.pat.pkl', 'rb') as f:
            #     common_pat[task] = pkl.load(f)
            #     f.close()

            if FLAGS.method == "mlp":
                with open(self.intmd_path + task + '.pos.mat.pkl', 'rb') as f:
                    X_pos_mat, y_pos_mat = pkl.load(f)
                    f.close()
                with open(self.intmd_path + task + '.neg.mat.pkl', 'rb') as f:
                    X_neg_mat, y_neg_mat = pkl.load(f)
                    f.close()

                # aggregate (and normalize) the data
                X_pos, y_pos = [], []
                X_neg, y_neg = [], []
                for s, array in X_pos_mat.items():
                    X_pos.append(np.sum(X_pos_mat[s], axis=0))
                    y_pos.append(y_pos_mat[s])

                for s, array in X_neg_mat.items():
                    X_neg.append(np.sum(X_neg_mat[s], axis=0))
                    y_neg.append(y_neg_mat[s])

                if FLAGS.meta_batch_size*FLAGS.update_batch_size > len(X_pos):
                    print ("length error!")

                self.data_to_show[task] = np.concatenate((X_pos[:(FLAGS.meta_batch_size*FLAGS.update_batch_size)], X_neg[:(FLAGS.meta_batch_size*FLAGS.update_batch_size)]), axis=0)
                self.label_to_show[task] = np.concatenate((y_pos[:(FLAGS.meta_batch_size*FLAGS.update_batch_size)], y_neg[:(FLAGS.meta_batch_size*FLAGS.update_batch_size)]), axis=0)

            elif FLAGS.method == "cnn" or FLAGS.method == "rnn":

                with open(self.intmd_path + task + '.pos.pkl', 'rb') as f:
                    X_pos_mat, y_pos_mat = pkl.load(f)
                    f.close()

                with open(self.intmd_path + task + '.neg.pkl', 'rb') as f:
                    X_neg_mat, y_neg_mat = pkl.load(f)
                    f.close()

                X_pos, y_pos = [], []
                X_neg, y_neg = [], []
                for s, array in X_pos_mat.items():
                     X_pos.append(array) # X_pos_mat[s] size: seq_len x n_words
                     y_pos.append(y_pos_mat[s])

                for s, array in X_neg_mat.items():
                     X_neg.append(array)
                     y_neg.append(y_neg_mat[s])

                X_pos, X_neg = self.get_fixed_timesteps(X_pos, X_neg)
                X_pos, X_neg = self.get_fixed_codesize(X_pos, X_neg)
                X_pos = self.get_feed_records(X_pos)
                X_neg = self.get_feed_records(X_neg)

                self.data_to_show[task] = np.concatenate((X_pos[:(FLAGS.meta_batch_size*FLAGS.update_batch_size)], X_neg[:(FLAGS.meta_batch_size*FLAGS.update_batch_size)]), axis=0)
                self.label_to_show[task] = np.concatenate((y_pos[:(FLAGS.meta_batch_size*FLAGS.update_batch_size)], y_neg[:(FLAGS.meta_batch_size*FLAGS.update_batch_size)]), axis=0)
            # print ('------')

    def load_data_vector(self, task):
        '''load aggregated data vectors for mlp. One vector per sample'''
        X_pos, y_pos = [], []
        X_neg, y_neg = [], []
        with open(self.intmd_path + task + '.pos.mat.pkl', 'rb') as f:
            X_pos_mat, y_pos_mat = pkl.load(f)
            f.close()

        with open(self.intmd_path + task + '.neg.mat.pkl', 'rb') as f:
            X_neg_mat, y_neg_mat = pkl.load(f)
            f.close()

        print ("The number of positive samles in task %s is: " %task, len(y_pos_mat))
        print ("The number of negative samles in task %s is: " %task, len(y_neg_mat))

        # aggregate (and normalize) the data
        for s, array in X_pos_mat.items():
            X_pos.append(np.sum(X_pos_mat[s], axis=0))
            y_pos.append(y_pos_mat[s])
        for s, array in X_neg_mat.items():
            X_neg.append(np.sum(X_neg_mat[s], axis=0))
            y_neg.append(y_neg_mat[s])
        # if self.pos_ratio is None:
        #     self.pos_ratio = int(len(y_pos)/len(y_neg))
        # return (X_pos, X_neg), (y_pos, y_neg)
        X, y = np.concatenate((X_pos, X_neg), axis=0), np.concatenate((y_pos, y_neg), axis=0)
        return X, y

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

        # n_codes_pos = []
        # n_codes_neg = []

        if self.target[0] == task and self.pat_reduce is True:
            counter = 0
            bounder = int(len(X_pos_mat) * self.ratio_t)

            for s, array in X_pos_mat.items():
                 X_pos.append(array) # X_pos_mat[s] size: seq_len x n_words
                 y_pos.append(y_pos_mat[s])
                 counter += 1
                 if counter == bounder:
                     print (counter)
                     break

            counter = 0
            bounder = int(len(X_neg_mat) * self.ratio_t)

            for s, array in X_neg_mat.items():
                 X_neg.append(array)
                 y_neg.append(y_neg_mat[s])

                 counter += 1
                 if counter == bounder:
                     break
        else:
            for s, array in X_pos_mat.items():
                 X_pos.append(array) # X_pos_mat[s] size: seq_len x n_words
                 y_pos.append(y_pos_mat[s])
                 # self.code_set=self.code_set.union(set([i for i in array[:].ravel() if i != PADDING_ID]))

            for s, array in X_neg_mat.items():
                 X_neg.append(array)
                 y_neg.append(y_neg_mat[s])
                 # self.code_set=self.code_set.union(set([i for i in array[:].ravel() if i != PADDING_ID]))

        # save code_size
        f = open(self.intmd_path + task + '.code.size.pkl', 'rb')
        self.task_code_size[task] = pkl.load(f)
        f.close()
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
        if FLAGS.method == "mlp":
            self.dim_input = [N_WORDS-1]
            for i in range(len(self.source)):
                data_s[i], label_s[i] = self.load_data_vector(self.source[i])
            for i in range(len(self.target)):
                data_t[i], label_t[i] = self.load_data_vector(self.target[i])

        elif FLAGS.method == "cnn" or "rnn":
            self.dim_input = [TIMESTEPS, N_WORDS]
            for i in range(len(self.source)):
                data_s[i], label_s[i] = self.load_data_matrix(self.source[i])

            for i in range(len(self.target)):
                data_t[i], label_t[i] = self.load_data_matrix(self.target[i])
        return data_s, data_t, label_s, label_t

    def generate_pretrain_data(self, is_training=True, ifold=0):
        ''' get pretraining samples and labels'''
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
                print (data_t.shape)
                print (label_t.shape)
                # print (len(label_t))
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
            # dump meta batches
            # if "metaval" in prefix:
            #     print ('write meta batch file ...')
            #     with open(self.intmd_path + "meta.batch." + prefix + ".pkl", 'wb') as f:
            #         pkl.dump((sample, label),f,protocol=2)
            #         f.close()
            #     print ('file written!')
        print ("batch counts: ", batch_count)
        sample = np.array(sample, dtype="float32")
        label = np.array(label, dtype="float32")
        print (sample.shape)
        print (label.shape)
        return sample, label

    def generate_meta_idx_batches(self, is_training=True, ifold=0, is_represent=False, episode_name=None):
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

        if is_represent:
            for task in self.data_to_show:
                try:
                    data_s = self.data_s
                    label_s = self.label_s
                    data_t = self.data_to_show[task]
                    label_t = self.label_to_show[task]
                    self.n_total_batches = int(len(label_t)/self.n_samples_per_task)
                except:
                    print ("Error: split training and validate first!")

                # generate episode
                episode = episode_name
                # print (episode_name)
                s_dict, t_dict = dict(), dict()
                for i in range(len(self.source)):
                    s_dict[i] = range(len(self.label_s[i]))

                t_list = list(range(len(label_t)))
                random.shuffle(t_list)
                batch_count = 0
                for _ in tqdm.tqdm(range(self.n_total_batches), 'generating meta batches'): # progress bar
                    # i.e., sample 16 patients from selected tasks
                    # len of spl and lbl: 4 * 16
                    idx = [] # index in one episode
                    for i in range(len(self.source)): # fetch from source tasks olderly
                        s_idx = random.sample(s_dict[i], self.n_samples_per_task)
                        idx.extend(s_idx)
                    # t_idx = range(batch_count*self.n_samples_per_task, (batch_count+1)*self.n_samples_per_task)
                    t_idx = random.sample(t_list, self.n_samples_per_task)
                    idx.extend(t_idx)
                    t_list = list(set(t_list) - set(t_idx))
                    batch_count += 1
                    # add meta_batch
                    episode[task].append(idx)
                # print (episode[task])
        return episode


    def make_data_tensor(self, is_training=True):
        '''make data a tensor with its 1st dim as batch size'''
        if is_training:
            print ('Batch train samples')
            samples = self.sample
            labels = self.label
        else:
            print ('Batch validate samples')
            samples = self.sample_val
            labels = self.label_val

        samples_per_batch = self.n_tasks * self.n_samples_per_task # total samples for one meta batch (number of episodes = n_samples_per_task)
        batch_sample_size = self.meta_batch_size * samples_per_batch # number of samples in the all the meta batches

        # Batching samples
        print('Batching samples ...')
        # Create a dataset tensor from the samples and the labels
        dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
        dataset = dataset.repeat()  # Automatically refill the data queue when empty
        dataset = dataset.batch(self.meta_batch_size)  # Create batches of data
        dataset = dataset.prefetch(self.meta_batch_size) # Prefetch data for faster consumption

         # Create an iterator over the dataset
        iterator = dataset.make_initializable_iterator()
        sample_batch, label_batch = iterator.get_next()

        print('sample_batch:', sample_batch.get_shape())
        print('label_batch:', label_batch.get_shape())
        return sample_batch, label_batch

        # num_preprocess_threads = 1 # TODO - enable this to be set to >1
        # min_queue_examples = 3
        # index_batch = tf.train.batch(
        #               [indices],
        #               batch_size=self.meta_batch_size,
        #               num_threads=num_preprocess_threads,
        #               capacity=min_queue_examples + self.meta_batch_size,
        #               enqueue_many=True,
        #         )
        # print (index_batch)
        # sample_batches, label_batches = samples[index_batch], labels[index_batch]]
        # tensor_x = torch.stack([torch.Tensor(i) for i in samples]) # transform to torch tensors
        # tensor_y = torch.stack([torch.Tensor(i) for i in labels])
        # all_data = utils.TensorDataset(tensor_x, tensor_y)

        # batches = BatchLoader(all_data, batch_size=self.meta_batch_size, shuffle=False, num_workers=1, drop_last=True)
        # for sample_id, sample_batched in enumerate(batches):
        #     print (sample_id)
        #     print (sample_batched)
        #     sample_batch, label_batch = sample_batched
        #     print (sample_batch)
        #     print ('-----')
        # for sample_id, sample_batched in enumerate(self.get_dataloader(shuffle=True)):
        #     preprocess_sample = self.sample_iter_data(sample=sample_batched, num_gpus=self.dataset.num_of_gpus,
        #                                               samples_per_iter=self.batches_per_iter,
        #                                               batch_size=self.dataset.batch_size)
        #     yield preprocess_sample

        # print('Manipulating sample data to be right shape')
        ## assume self.meta_batch_size = 4, self.n_tasks=4, self_samples_per_task=16
        # all_sample_batches, all_label_batches = [], []
        # for i in range(self.meta_batch_size): # 4 meta batches
        #     sample_batch = samples[i : i+1] # length of samples equals to the batch_sample_size
        #     label_batch = labels[i : i+1]
        #     new_list, new_label_list = [], []
        #     # for each sample in all 4 tasks (from 0 to 3)
        #     for k in range(self.n_samples_per_task): # 16 episodes
        #         task_idxs = tf.range(0, self.n_tasks)
        #         task_idxs = tf.concat([tf.random_shuffle(task_idxs[:-1]), tf.convert_to_tensor([task_idxs[-1]]).get_shape()], 0) # fix the last task as target
        #         # possible problem: all the 4 tasks in the episode is positive or negative
        #         true_idxs = task_idxs * self.n_samples_per_task + k # true index in sample_batch
        #         new_list.append(tf.gather(sample_batch, true_idxs)) # Gather slices from params axis according to indices
        #         new_label_list.append(tf.gather(label_batch, true_idxs))
        #     new_list = tf.concat(new_list, 0)  # has shape [self.n_tasks*self.n_samples_per_task, self.dim_input], [4*16, 1016] or [4*16, timesteps*code_size]
        #     new_label_list = tf.concat(new_label_list, 0) # [4*16]
        #     all_sample_batches.append(new_list)
        #     all_label_batches.append(new_label_list)
        #
        # all_sample_batches = tf.stack(all_sample_batches) # [4, 4*16, 1016] or [4, 80, timesteps*code_size]
        # all_label_batches = tf.stack(all_label_batches)
        # print('sample_batch:', sample_batch.get_shape())
        # print('label_batch:', label_batch.get_shape())
        # return sample_batch, label_batch
