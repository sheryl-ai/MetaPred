from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers


import sklearn
import numpy as np
import os, time, shutil, collections

PADDING_ID = 1016
WORDS_NUM = 1017
MASK_ARRAY = [[1.]] * PADDING_ID + [[0.]] + [[1.]] * (WORDS_NUM - PADDING_ID - 1)

class BaseModel(object):
    """
    Base Model for basic networks with sequential data, i.e., RNN, CNN.
    """
    def __init__(self):
        self.regularizers = []

    def loss(self, logits):
        # Define loss and optimizer
        with tf.name_scope('cross_entropy'):
            labels = tf.to_int64(self.ph_labels)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            cross_entropy = tf.reduce_mean(cross_entropy)
        with tf.name_scope('regularization'):
            regularization = self.regularization
            regularization *= tf.add_n(self.regularizers)
        loss = cross_entropy + regularization

        # Summaries for TensorBoard.
        tf.summary.scalar('loss/cross_entropy', cross_entropy)
        tf.summary.scalar('loss/regularization', regularization)
        tf.summary.scalar('loss/total', loss)
        with tf.name_scope('averages'):
            averages = tf.train.ExponentialMovingAverage(0.9)
            op_averages = averages.apply([cross_entropy, regularization, loss])
            tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
            tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
            tf.summary.scalar('loss/avg/total', averages.average(loss))
            with tf.control_dependencies([op_averages]):
                loss_average = tf.identity(averages.average(loss), name='control')
        return loss, loss_average

    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end, :, :]

            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1, self.ph_training: False}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end-begin]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
        return op_train

    def fit(self, X_tr, y_tr, X_vl, y_vl):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)

        # Training.
        count = 0
        bad_counter = 0
        accuracies = []
        aucs = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * X_tr.shape[0] / self.batch_size)
        estop = False  # early stop
        if type(X_vl) is not np.ndarray:
            X_vl = X_vl.toarray()

        for step in range(1, num_steps+1):
            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(X_tr.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            count += len(idx)
            batch_data, batch_labels = X_tr[idx, :, :], y_tr[idx]

            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout, self.ph_training: True}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                print ('Seen samples: %d' % count)
                epoch = step * self.batch_size / X_tr.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, auc, accuracy, loss, predictions = self.evaluate(X_vl, y_vl, sess)
                aucs.append(auc)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                # print(predictions.tolist()[:50])
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validataion/auc', simple_value=auc)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

                if len(aucs) > (self.patience+5) and auc > np.array(aucs).max():
                    bad_counter = 0

                if len(aucs) > (self.patience+5) and auc <= np.array(aucs)[:-self.patience].max():
                    bad_counter += 1
                    if bad_counter > self.patience:
                        print('Early Stop!')
                        estop = True
                        break
            if estop:
                break
        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        print('validation auc: peak = {:.2f}, mean = {:.2f}'.format(max(aucs), np.mean(aucs[-10:])))
        writer.close()
        sess.close()
        t_step = (time.time() - t_wall) / num_steps
        print ("Optimization Finished!")
        return  aucs, accuracies, losses

    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.
        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        """
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)

        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
        auc = 100 * sklearn.metrics.auc(fpr, tpr)
        ncorrects = sum(predictions == labels)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        string = 'auc: {:.2f}, accuracy: {:.2f} ({:d} / {:d}), loss: {:.2e}'.format(auc, accuracy, ncorrects, len(labels), loss)

        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        # return string, auc, loss, predictions
        return string, auc, accuracy, loss, predictions


    def inference(self, data, dropout, is_training):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, dropout, is_training)
        return logits

    def _weight_variable(self, shape):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if self.isReg:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if self.isReg:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout])
        b = self._bias_variable([Mout])
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def normalize(self, inputs, epsilon = 1e-8, scope="ln", reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
        return outputs

    # Helper methods.
    def _get_path(self, folder):
        path = '../../models/'
        return os.path.join(path, folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _get_prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    # Methods to construct the computational graph
    def build_model(self):
        """Build the computational graph with memory network of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                # tf Graph input
                self.ph_data = tf.placeholder(tf.int32, (self.batch_size, self.timesteps, self.code_size), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.ph_training = tf.placeholder(tf.bool, name='trainingFlag')

            # Construct model
            op_logits, self.op_represent = self.inference(self.ph_data, self.ph_dropout, self.ph_training)
            self.op_loss, self.op_loss_average = self.loss(op_logits)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self._get_prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        self.graph.finalize()


class vrnn(BaseModel):
    """
    Build a vanilla recurrent neural network.
    """
    def __init__(self, n_words, n_classes, timesteps, code_size, dir_name, init_std=0.05):
        super().__init__()
        # training parameters
        self.learning_rate = 0.05
        self.batch_size = 64
        self.num_epochs = 200
        self.dropout = 0.8
        self.decay_rate = 0.9
        self.decay_steps = 10000 / self.batch_size
        self.momentum = 0.95
        self.patience = 10
        self.eval_frequency = self.num_epochs
        self.regularization = 0.01
        self.isReg = True
        self.dir_name =  dir_name

        # Network Parameters
        self.init_std = init_std
        self.n_hidden = 256 # hidden dimensions of embedding
        self.n_hidden_1 = 128
        self.n_hidden_2 = 128
        self.n_words = n_words
        self.n_classes = n_classes
        self.timesteps = timesteps
        self.code_size = code_size
        self.M = [self.n_hidden_1, self.n_classes]
        self.build_model()

    def build_emb(self, x):
        self.Wemb = tf.Variable(tf.random_normal([self.n_words, self.n_hidden], stddev=self.init_std))
        self.Wemb_mask = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)

        _x = tf.nn.embedding_lookup(self.Wemb, x) # recs size is (batch_size, mem_size, n_words)
        _x_mask = tf.nn.embedding_lookup(self.Wemb_mask, x)
        emb_vecs = tf.multiply(_x, _x_mask) # broadcast
        emb_vecs = tf.reduce_sum(emb_vecs, 2)
        return emb_vecs

    def lstm(self, x):
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        # x = tf.unstack(x, self.timesteps, 1)
        # lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0) # Define a lstm cell with tensorflow
        # h, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        # print (h[-1].get_shape())
        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        output, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
        output_sum = tf.reduce_sum(output, axis=1)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        return last

    def gru(self, x):
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, self.timesteps, 1)
        gru_cell = rnn.GRUCell(self.n_hidden) # Define a gru cell with tensorflow
        h, states = rnn.static_rnn(gru_cell, x, dtype=tf.float32)
        return h[-1]

    def build_attention(self, x, output_size, initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
        '''similar to the method in Hierarchical Attention Networks for Document Classification'''
        assert len(x.get_shape()) == 3 and x.get_shape()[-1].value is not None

        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        x_projection = layers.fully_connected(x, output_size,
                                              activation_fn=activation_fn,
                                              scope=scope)

        vector_attn = tf.reduce_sum(tf.multiply(x_projection, attention_context_vector), axis=2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(x_projection, attention_weights)
        outputs = tf.reduce_sum(weighted_projection, axis=1)
        return outputs

    # Create model
    def _inference(self, x, dropout, is_training=True):
        # embedding
        with tf.variable_scope("embedding"):
            x = self.build_emb(x)
            x = self.normalize(x)

        # recurrent neural networks
        with tf.variable_scope("rnn"):
            # hout = self.gru(x)
            hout = self.lstm(x)

        with tf.variable_scope("dropout"):
            h_ = layers.dropout(hout, keep_prob=dropout)

        # fully connected layers
        for i, dim in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                h_ = self.fc(h_, dim)
                h_ = tf.nn.dropout(h_, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            prob = self.fc(h_, self.M[-1], relu=False)
        return prob


class birnn(BaseModel):
    def __init__(self, n_words, n_classes, timesteps, code_size, dir_name, init_std=0.05):
        super().__init__()
        # training parameters
        self.learning_rate = 0.05
        self.batch_size = 64
        self.num_epochs = 200
        self.dropout = 0.8
        self.decay_rate = 0.9
        self.decay_steps = 10000 / self.batch_size
        self.momentum = 0.95
        self.patience = 10
        self.eval_frequency = self.num_epochs
        self.regularization = 0.01
        self.isReg = True
        self.dir_name =  dir_name

        # Network Parameters
        self.init_std = init_std
        self.n_hidden = 256 # hidden dimensions of embedding
        self.n_hidden_1 = 128
        self.n_hidden_2 = 128
        self.n_words = n_words
        self.n_classes = n_classes
        self.timesteps = timesteps
        self.code_size = code_size
        self.M = [self.n_hidden_1, self.n_classes]
        self.build_model()

    def build_emb(self, x):
        with tf.variable_scope("embed"):
            self.Wemb = tf.Variable(tf.random_normal([self.n_words, self.n_hidden], stddev=self.init_std))
            self.Wemb_mask = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)

            _x = tf.nn.embedding_lookup(self.Wemb, x) # recs size is (batch_size, mem_size, n_words)
            _x_mask = tf.nn.embedding_lookup(self.Wemb_mask, x)
            emb_vecs = tf.multiply(_x, _x_mask) # broadcast
            emb_vecs = tf.reduce_sum(emb_vecs, 2)
        return emb_vecs

    def bilstm(self, x):
        x = tf.unstack(x, self.timesteps, 1)

        with tf.variable_scope('birnn') as scope:
            with tf.variable_scope('forward'):
                lstm_fw_cell = rnn.BasicLSTMCell(int(self.n_hidden/2), forget_bias=1.0)
            # Backward direction cell
            with tf.variable_scope('backward'):
                lstm_bw_cell = rnn.BasicLSTMCell(int(self.n_hidden/2), forget_bias=1.0)
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                  dtype=tf.float32)
        except Exception: # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                    dtype=tf.float32)
        return outputs[-1]

    # Create model
    def _inference(self, x, dropout, is_training=True):
        # embedding
        with tf.variable_scope("embedding"):
            x = self.build_emb(x)
            x = self.normalize(x)

        # recurrent neural networks
        with tf.variable_scope("birnn"):
            # hout = self.gru(x)
            hout = self.bilstm(x)

        with tf.variable_scope("dropout"):
            h_ = layers.dropout(hout, keep_prob=dropout)

        # fully connected layers
        for i, dim in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                h_ = self.fc(h_, dim)
                h_ = tf.nn.dropout(h_, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            prob = self.fc(h_, self.M[-1], relu=False)
        return prob


class cnn(BaseModel):
    def __init__(self, n_words, n_classes, timesteps, code_size, dir_name, init_std=0.05):
        super().__init__()
        # training parameters
        self.learning_rate = 0.01
        self.batch_size = 32
        self.num_epochs = 200
        self.dropout = 0.6
        self.decay_rate = 0.9
        self.decay_steps = 10000 / self.batch_size
        self.momentum = 0.95
        self.patience = 10
        self.eval_frequency = self.num_epochs
        self.regularization = 0.01
        self.isReg = True
        self.dir_name =  dir_name

        # Network Parameters
        self.init_std = init_std
        self.n_hidden = 256 # hidden dimensions of embedding
        self.n_hidden_1 = 128
        self.n_hidden_2 = 128
        self.n_words = n_words
        self.n_classes = n_classes
        self.n_filters = 128
        self.timesteps = timesteps
        self.code_size = code_size
        self.M = [self.n_hidden_1, self.n_classes]
        self.filter_sizes = [3, 4, 5]
        self.build_model()

    def build_emb(self, x):
        with tf.variable_scope("embed"):
            self.Wemb = tf.Variable(tf.random_normal([self.n_words, self.n_hidden], stddev=self.init_std))
            self.Wemb_mask = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)

            _x = tf.nn.embedding_lookup(self.Wemb, x) # recs size is (batch_size, mem_size, n_words)
            _x_mask = tf.nn.embedding_lookup(self.Wemb_mask, x)
            emb_vecs = tf.multiply(_x, _x_mask) # broadcast
            emb_vecs = tf.reduce_sum(emb_vecs, 2)
            self.emb_expanded = tf.expand_dims(emb_vecs, -1)
        return emb_vecs

    def build_conv(self, x, is_training):
        '''Create a convolution + maxpool layer for each filter size'''
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.n_hidden, 1, self.n_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.n_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.emb_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.leaky_relu(tf.nn.bias_add(conv, b), name="relu")
                h = layers.batch_norm(h, updates_collections=None,
                                         decay=0.99,
                                         scale=True, center=True,
                                         is_training=is_training)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.timesteps - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.n_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat

    # Create model
    def _inference(self, x, dropout, is_training=True):
        with tf.variable_scope("embedding"):
            xemb = self.build_emb(x)

        # convolutional network
        with tf.variable_scope("conv"):
            hout = self.build_conv(xemb, is_training)

        with tf.variable_scope("dropout"):
            h_ = layers.dropout(hout, keep_prob=dropout)

        for i, dim in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                h_ = self.fc(h_, dim)
                h_ = tf.nn.dropout(h_, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            prob = self.fc(h_, self.M[-1], relu=False)
        return prob
