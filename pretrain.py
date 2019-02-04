""" Code for the MAML algorithm and network architecture. """
import numpy as np
import sklearn
import tensorflow as tf
import os, time, shutil, collections

from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

PADDING_ID = 1016
WORDS_NUM = 1017
MASK_ARRAY = [[1.]] * PADDING_ID + [[0.]] + [[1.]] * (WORDS_NUM - PADDING_ID - 1)

SUMMARY_INTERVAL = 100
SAVE_INTERVAL = 1000
PRINT_INTERVAL = 100
TEST_PRINT_INTERVAL = PRINT_INTERVAL*5


class BaseModel(object):
    """
    Base Model for basic networks with sequential data, i.e., RNN, CNN.
    """
    def __init__(self):
        self.regularizers = []
        self.regularization = 0.01
        self.isReg = True


    def evaluate(self, data, labels, sess=None, prefix="metatest_"):
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

            if FLAGS.method == "mlp":
                batch_data, batch_labels = X_tr[idx, :], y_tr[idx]
                if type(batch_data) is not np.ndarray:
                    batch_data = batch_data.toarray()  # convert sparse matrices
                if self.is_finetune:
                    feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: 1}
                else:
                    feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}

            elif FLAGS.method == "rnn" or FLAGS.method == "cnn":
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

        # store weights value for fine-tune
        if self.is_finetune is not True:
            feed_dict = {}
            for k in self.op_weights:
                self.weights_for_init[k] = sess.run([self.op_weights[k]], feed_dict)[0]
                self.weights_for_finetune[k] = sess.run([self.op_weights[k]], feed_dict)[0]

        writer.close()
        sess.close()
        t_step = (time.time() - t_wall) / num_steps

        return  sess, aucs, accuracies

    def loss(self, logits):
        # Define loss and optimizer
        with tf.name_scope('cross_entropy'):
            labels = tf.to_int64(self.ph_labels)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            cross_entropy = tf.reduce_mean(cross_entropy)
        if self.is_finetune and self.freeze_opt == 'mlp':
            loss = cross_entropy
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
        else:
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
            if FLAGS.method == "mlp":
                batch_data = np.zeros((self.batch_size, data.shape[1]))
                tmp_data = data[begin:end, :]

                if type(tmp_data) is not np.ndarray:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                batch_data[:end-begin] = tmp_data
                feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}

            elif FLAGS.method == "rnn" or FLAGS.method == "cnn":
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

    # def weight_variable(self, shape, name='weights'):
    #     initial = tf.truncated_normal_initializer(0, 0.1)
    #     var = tf.get_variable(name, shape, tf.float32, initializer=initial)
    #
    #     if self.isReg:
    #         self.regularizers.append(tf.nn.l2_loss(var))
    #     tf.summary.histogram(var.op.name, var)
    #     return var
    #
    # def bias_variable(self, shape, name='bias'):
    #     initial = tf.constant_initializer(0.1)
    #     var = tf.get_variable(name, shape, tf.float32, initializer=initial)
    #
    #     if self.isReg:
    #         self.regularizers.append(tf.nn.l2_loss(var))
    #     tf.summary.histogram(var.op.name, var)
    #     return var

    def weight_variable(self, shape, name='weights'):
        if self.is_finetune:
            print("==")
            initial = self.finetune_weights[name]
            var = tf.Variable(initial_value=initial, name=name)
        else:
            initial = tf.truncated_normal_initializer(0, 0.1)
            var = tf.get_variable(name, shape, tf.float32, initializer=initial)

        if self.isReg:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def bias_variable(self, shape, name='bias'):
        if self.is_finetune:
            print("==")
            initial = self.finetune_weights[name]
            var = tf.Variable(initial_value=initial, name=name)
        else:
            initial = tf.constant_initializer(0.1)
            var = tf.get_variable(name, shape, tf.float32, initializer=initial)

        if self.isReg:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def build_fc_weights(self, dim_in, weights):
        for i, dim in enumerate(self.dim_hidden):
            dim_out = dim
            weights["fc_W"+str(i)] = self.weight_variable([int(dim_in), dim_out], name="fc_W"+str(i))
            weights["fc_b"+str(i)] = self.bias_variable([dim_out], name="fc_b"+str(i))
            dim_in = dim_out
        return weights

    def fc(self, x, W, b, relu=True):
        """Fully connected layer with Mout features."""
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


class MLP(BaseModel):
    """
    Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron).
    """
    def __init__(self, data_loader, finetune_m, freeze_opt=None, is_finetune=False):
        super().__init__()
        self.is_finetune = is_finetune
        self.freeze_opt = freeze_opt
        print ("freeze_opt: ", self.freeze_opt)
        if self.is_finetune:
            self.finetune_weights = finetune_m.weights_for_finetune

        # Training Parameters
        self.dir_name =  "mlp"
        self.learning_rate = 0.1
        self.batch_size = 64
        self.num_epochs = 200
        self.dropout = 0.8
        self.decay_rate = 0.9
        self.decay_steps = 5000/ self.batch_size
        self.momentum = 0.95
        self.patience = 5
        self.eval_frequency = self.num_epochs

        # Network Parameters
        self.n_hidden_1 = 128 # 1st layer number of neurons
        self.n_hidden_2 = 128 # 2nd layer number of neurons
        self.num_input =  data_loader.dim_input[0]
        self.num_classes = FLAGS.n_classes
        self.dim_hidden = [self.n_hidden_1, self.n_hidden_2, self.num_classes]

        self.weights_for_init = dict() # to store the value of learned params
        self.weights_for_finetune = dict()

        print('method', self.dir_name, 'data shape:', [self.num_input], 'batch size:', self.batch_size, 'learning rate:', self.learning_rate, \
              'momentum:', self.momentum, 'patience:', self.patience)

        self.build_model()

    # Methods to construct the computational graph with mlp.
    def build_model(self):
        """Build the computational graph with memory network of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                # tf Graph input
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, self.num_input), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Construct model
            op_logits, self.weights = self._inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self._get_prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            self.op_weights = self.get_op_variables()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        self.graph.finalize()

    def get_op_variables(self):
        op_weights = dict()
        op_var = tf.trainable_variables()
        for i, dim in enumerate(self.dim_hidden):
            op_weights["fc_W"+str(i)] = [v for v in op_var if "fc_W"+str(i) in v.name ][0]
            op_weights["fc_b"+str(i)] = [v for v in op_var if "fc_b"+str(i) in v.name][0]
        return op_weights

    # Create model
    def _inference(self, x, dropout):
        with tf.variable_scope('model', reuse=None) as training_scope:
            weights = {}
            _, dim_in = x.get_shape()
            weights = self.build_fc_weights(dim_in, weights)

            for i, dim in enumerate(self.dim_hidden[:-1]):
                x = self.fc(x, weights["fc_W"+str(i)], weights["fc_b"+str(i)])
                x = tf.nn.dropout(x, dropout)

            # Logits linear layer, i.e. softmax without normalization.
            N, Min = x.get_shape()
            i = len(self.dim_hidden)-1
            logits = self.fc(x, weights["fc_W"+str(i)], weights["fc_b"+str(i)], relu=False)
            return logits, weights

class RNN(BaseModel):
    """
    Build a vanilla recurrent neural network.
    """
    def __init__(self, data_loader, finetune_m=None, init_std=0.05, freeze_opt=None, is_finetune=False):
        super().__init__()
        self.is_finetune = is_finetune
        self.freeze_opt = freeze_opt
        print ("freeze_opt: ", self.freeze_opt)
        if self.is_finetune:
            print ("==")
            self.finetune_weights = finetune_m.weights_for_finetune
            self.learning_rate = 0.00001
            self.batch_size = 128
            self.num_epochs = 30
        else:
            self.learning_rate = 0.5
            self.batch_size = 128
            self.num_epochs = 200

        # training parameters
        self.dir_name =  "rnn"
        self.dropout = 1
        self.decay_rate = 0.9
        self.decay_steps = 10000 / self.batch_size
        self.momentum = 0.95
        self.patience = 5
        self.eval_frequency = self.num_epochs

        # Network Parameters
        self.init_std = init_std
        self.n_hidden = 256 # hidden dimensions of embedding
        self.n_hidden_1 = 128
        self.n_hidden_2 = 128
        self.n_words = data_loader.n_words
        self.num_input = data_loader.dim_input
        self.n_classes = FLAGS.n_classes
        self.timesteps = data_loader.timesteps
        self.code_size = data_loader.code_size
        self.dim_hidden = [self.n_hidden_1, self.n_hidden_2, FLAGS.n_classes]

        self.weights_for_init = dict() # to store the value of learned params
        self.weights_for_finetune = dict()

        self.build_model()

        print('method', self.dir_name, 'data shape:', self.num_input, 'batch size:', self.batch_size, 'learning rate:', self.learning_rate, \
              'momentum:', self.momentum, 'patience:', self.patience)

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
            op_logits = self._inference(self.ph_data, self.ph_dropout, self.ph_training)
            self.op_loss, self.op_loss_average = self.loss(op_logits)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self._get_prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            if self.is_finetune is not True:
                self.op_weights = self.get_op_variables()
            else:
                print (tf.trainable_variables())

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        self.graph.finalize()

    def get_op_variables(self):
        op_weights = dict()
        op_var = tf.trainable_variables()

        # embedding
        op_weights["emb_W"] = [v for v in op_var if "emb_W" in v.name][0]
        # lstm
        op_weights["lstm_W_xh"] = [v for v in op_var if "lstm_W_xh" in v.name][0]
        op_weights["lstm_W_hh"] = [v for v in op_var if "lstm_W_hh" in v.name][0]
        op_weights["lstm_b"] = [v for v in op_var if "lstm_b" in v.name][0]
        # fully connected
        for i, dim in enumerate(self.dim_hidden):
            op_weights["fc_W"+str(i)] = [v for v in op_var if "fc_W"+str(i) in v.name ][0]
            op_weights["fc_b"+str(i)] = [v for v in op_var if "fc_b"+str(i) in v.name][0]
        print ('show variable')
        print(op_var)
        return op_weights


    def build_emb_weights(self, weights):
        weights["emb_W"] = tf.Variable(tf.random_normal([self.n_words, self.n_hidden], stddev=self.init_std), name="emb_W")
        weights["emb_mask_W"] = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)
        return weights

    def embedding(self, x, Wemb, Wemb_mask):
        _x = tf.nn.embedding_lookup(Wemb, x) # recs size is (batch_size, timesteps, n_words)
        _x_mask = tf.nn.embedding_lookup(Wemb_mask, x)
        emb_vecs = tf.multiply(_x, _x_mask) # broadcast
        emb_vecs = tf.reduce_sum(emb_vecs, 2)
        return emb_vecs

    def lstm_identity_initializer(self, scale):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            """Ugly cause LSTM params calculated in one matrix multiply"""
            size = shape[0]
            # gate (j) is identity
            t = np.zeros(shape)
            t[:, size:size * 2] = np.identity(size) * scale
            t[:, :size] = self.orthogonal([size, size])
            t[:, size * 2:size * 3] = self.orthogonal([size, size])
            t[:, size * 3:] = self.orthogonal([size, size])
            return tf.constant(t, dtype=dtype)
        return _initializer

    def orthogonal_initializer(self):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            return tf.constant(self.orthogonal(shape), dtype)
        return _initializer

    def orthogonal(self, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)

    def build_lstm_weights(self, weights):
        #
        # # Keep W_xh and W_hh separate here as well to reuse initialization methods
        # with tf.variable_scope(scope or type(self).__name__):
        weights["lstm_W_xh"] = tf.get_variable('lstm_W_xh', [self.n_hidden, 4 * self.n_hidden],
                               initializer=self.orthogonal_initializer())
        weights["lstm_W_hh"] = tf.get_variable('lstm_W_hh', [self.n_hidden, 4 * self.n_hidden],
                               initializer=self.lstm_identity_initializer(0.95),)
        weights["lstm_b"] = tf.get_variable('lstm_b', [4 * self.n_hidden])
        return weights

    # def lstm(self, x):
    #     lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
    #     output, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    #     output = tf.transpose(output, [1, 0, 2])
    #     last = tf.gather(output, int(output.get_shape()[0]) - 1)
    #     return last


    # Create model
    def _inference(self, x, dropout, is_training=True):
        with tf.variable_scope('pretrain_model', reuse=None) as training_scope:
            if self.freeze_opt == None:
                weights = {}
                weights = self.build_emb_weights(weights)
                weights = self.build_lstm_weights(weights)
                weights = self.build_fc_weights(self.n_hidden, weights)

                # embedding
                with tf.variable_scope("embedding"):
                    xemb = self.embedding(x, weights["emb_W"], weights["emb_mask_W"])

                # recurrent neural networks
                with tf.variable_scope("rnn"):
                    lstm_cell = LSTMCell(self.n_hidden, weights["lstm_W_xh"], weights["lstm_W_hh"], weights["lstm_b"])
                    # lstm_cell = LSTMCell(self.n_hidden)
                    xemb = tf.unstack(xemb, self.timesteps, 1)

                    #c, h
                    W_state_c = tf.random_normal([self.batch_size, self.n_hidden], stddev=0.1)
                    W_state_h = tf.random_normal([self.batch_size, self.n_hidden], stddev=0.1)
                    outputs, state = tf.nn.static_rnn(lstm_cell, xemb, initial_state=(W_state_c, W_state_h), dtype=tf.float32)
                    _, hout = state

                with tf.variable_scope("dropout"):
                    h_ = layers.dropout(hout, keep_prob=dropout)

                for i, dim in enumerate(self.dim_hidden[:-1]):
                    h_ = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)])
                    h_ = tf.nn.dropout(h_, dropout)

                # Logits linear layer, i.e. softmax without normalization.
                N, Min = h_.get_shape()
                i = len(self.dim_hidden)-1
                logits = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)], relu=False)

            else:
                with tf.variable_scope("embedding"):
                    Wemb = self.finetune_weights["emb_W"]
                    Wemb_mask = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)
                    xemb = self.embedding(x, Wemb, Wemb_mask)


                # convolutional network
                with tf.variable_scope("rnn"):
                    lstm_cell = LSTMCell(self.n_hidden, self.finetune_weights["lstm_W_xh"], self.finetune_weights["lstm_W_hh"], self.finetune_weights["lstm_b"])
                    xemb = tf.unstack(xemb, self.timesteps, 1)
                    W_state_c = tf.random_normal([self.batch_size, self.n_hidden], stddev=0.1)
                    W_state_h = tf.random_normal([self.batch_size, self.n_hidden], stddev=0.1)
                    outputs, state = tf.nn.static_rnn(lstm_cell, xemb, initial_state=(W_state_c, W_state_h), dtype=tf.float32)
                    _, hout = state

                with tf.variable_scope("dropout"):
                    h_ = layers.dropout(hout, keep_prob=dropout)

                for i, dim in enumerate(self.dim_hidden[:-1]):
                    Wfc = self.finetune_weights["fc_W"+str(i)]
                    bfc = self.finetune_weights["fc_b"+str(i)]
                    h_ = self.fc(h_, Wfc, bfc)
                    h_ = tf.nn.dropout(h_, dropout)

                i = len(self.dim_hidden)-1
                weights = {}
                dim_in = self.n_hidden_2
                weights["fc_W"+str(i)] = self.weight_variable([int(dim_in), FLAGS.n_classes], name="fc_W"+str(i))
                weights["fc_b"+str(i)] = self.bias_variable([FLAGS.n_classes], name="fc_b"+str(i))

                # Logits linear layer, i.e. softmax without normalization.
                N, Min = h_.get_shape()
                i = len(self.dim_hidden)-1
                logits = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)], relu=False)
        return logits

# class LSTMCell(RNNCell):
#     """Vanilla LSTM implemented with same initializations as BN-LSTM"""
#     def __init__(self, num_units):
#         self.num_units = num_units
#
#     @property
#     def state_size(self):
#         return (self.num_units, self.num_units)
#
#     @property
#     def output_size(self):
#         return self.num_units
#
#     def lstm_identity_initializer(self, scale):
#         def _initializer(shape, dtype=tf.float32, partition_info=None):
#             """Ugly cause LSTM params calculated in one matrix multiply"""
#             size = shape[0]
#             # gate (j) is identity
#             t = np.zeros(shape)
#             t[:, size:size * 2] = np.identity(size) * scale
#             t[:, :size] = self.orthogonal([size, size])
#             t[:, size * 2:size * 3] = self.orthogonal([size, size])
#             t[:, size * 3:] = self.orthogonal([size, size])
#             return tf.constant(t, dtype=dtype)
#         return _initializer
#
#     def orthogonal_initializer(self):
#         def _initializer(shape, dtype=tf.float32, partition_info=None):
#             return tf.constant(self.orthogonal(shape), dtype)
#         return _initializer
#
#     def orthogonal(self, shape):
#         flat_shape = (shape[0], np.prod(shape[1:]))
#         a = np.random.normal(0.0, 1.0, flat_shape)
#         u, _, v = np.linalg.svd(a, full_matrices=False)
#         q = u if u.shape == flat_shape else v
#         return q.reshape(shape)
#
#     def __call__(self, x, state, scope=None):
#         with tf.variable_scope(scope or type(self).__name__):
#             c, h = state
#
#             # Keep W_xh and W_hh separate here as well to reuse initialization methods
#             x_size = x.get_shape().as_list()[1]
#             W_xh = tf.get_variable('W_xh',
#                 [x_size, 4 * self.num_units],
#                 initializer=self.orthogonal_initializer())
#             W_hh = tf.get_variable('W_hh',
#                 [self.num_units, 4 * self.num_units],
#                 initializer=self.lstm_identity_initializer(0.95))
#             bias = tf.get_variable('bias', [4 * self.num_units])
#             print (W_xh.get_shape())
#             print (W_hh.get_shape())
#             print (bias.get_shape())
#
#             # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
#             # improve speed by concat.
#             concat = tf.concat([x, h], 1)
#             W_both = tf.concat([W_xh, W_hh], 0)
#             hidden = tf.matmul(concat, W_both) + bias
#
#             i, j, f, o = tf.split(hidden, 4, axis=1)
#
#             new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
#             new_h = tf.tanh(new_c) * tf.sigmoid(o)
#
#             return new_h, (new_c, new_h)


class LSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''
    def __init__(self, num_units, W_xh, W_hh, bias):
        self.num_units = num_units
        self.W_xh = W_xh
        self.W_hh = W_hh
        self.bias = bias

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=tf.AUTO_REUSE):
            c, h = state

            # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
            # improve speed by concat.
            concat = tf.concat([x, h], 1)
            W_both = tf.concat([self.W_xh, self.W_hh], 0)
            hidden = tf.matmul(concat, W_both) + self.bias

            i, j, f, o = tf.split(hidden, 4, axis=1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)


class CNN(BaseModel):
    """
    Build a convolutional neural network.
    """
    def __init__(self, data_loader, finetune_m=None, init_std=0.05, freeze_opt=None, is_finetune=False):
        super().__init__()
        self.is_finetune = is_finetune
        self.freeze_opt = freeze_opt
        print ("freeze_opt: ", self.freeze_opt)
        if self.is_finetune:
            print ("==")
            self.finetune_weights = finetune_m.weights_for_finetune
            self.learning_rate = 0.00001
            self.batch_size = 64
            self.num_epochs = 30
        else:
            self.learning_rate = 0.1
            self.batch_size = 128
            self.num_epochs = 200

        # training parameters
        self.dir_name =  "cnn"


        self.dropout = 0.6
        self.decay_rate = 0.9
        self.decay_steps = 10000 / self.batch_size
        self.momentum = 0.95
        self.patience = 10
        self.eval_frequency = self.num_epochs

        # Network Parameters
        self.init_std = init_std
        self.n_hidden = 256 # hidden dimensions of embedding
        self.n_hidden_1 = 128
        self.n_hidden_2 = 128
        self.n_words = data_loader.n_words
        self.n_classes = FLAGS.n_classes
        self.n_filters = 128
        self.num_input = data_loader.dim_input
        self.timesteps = data_loader.timesteps
        self.code_size = data_loader.code_size
        self.dim_hidden = [self.n_hidden_1, self.n_hidden_2, FLAGS.n_classes]
        self.filter_sizes = [3, 4, 5]

        self.weights_for_init = dict() # to store the value of learned params
        self.weights_for_finetune = dict()

        print('method', self.dir_name, 'data shape:', self.num_input, 'batch size:', self.batch_size, 'learning rate:', self.learning_rate, \
              'momentum:', self.momentum, 'patience:', self.patience)
        self.build_model()

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
            op_logits = self._inference(self.ph_data, self.ph_dropout, self.ph_training)
            self.op_loss, self.op_loss_average = self.loss(op_logits)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self._get_prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            if self.is_finetune is not True:
                self.op_weights = self.get_op_variables()
            else:
                print (tf.trainable_variables())

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        self.graph.finalize()

    def get_op_variables(self):
        op_weights = dict()
        op_var = tf.trainable_variables()
        # embedding
        op_weights["emb_W"] = [v for v in op_var if "emb_W" in v.name][0]
        # cnn
        for i, filter_size in enumerate(self.filter_sizes):
            op_weights["conv_W"+str(filter_size)] = [v for v in op_var if "conv_W"+str(filter_size) in v.name][0]
            op_weights["conv_b"+str(filter_size)] = [v for v in op_var if "conv_b"+str(filter_size) in v.name][0]
        # fully connected
        for i, dim in enumerate(self.dim_hidden):
            op_weights["fc_W"+str(i)] = [v for v in op_var if "fc_W"+str(i) in v.name][0]
            op_weights["fc_b"+str(i)] = [v for v in op_var if "fc_b"+str(i) in v.name][0]
        return op_weights

    def build_emb_weights(self, weights):
        weights["emb_W"] = tf.Variable(tf.random_normal([self.n_words, self.n_hidden], stddev=self.init_std), name="emb_W")
        weights["emb_mask_W"] = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)
        return weights

    def embedding(self, x, Wemb, Wemb_mask):
        _x = tf.nn.embedding_lookup(Wemb, x) # recs size is (batch_size, timesteps, n_words)
        _x_mask = tf.nn.embedding_lookup(Wemb_mask, x)
        emb_vecs = tf.multiply(_x, _x_mask) # broadcast
        emb_vecs = tf.reduce_sum(emb_vecs, 2)
        self.emb_expanded = tf.expand_dims(emb_vecs, -1)
        return emb_vecs

    def build_conv_weights(self, weights):
        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.n_hidden, 1, self.n_filters]
            weights["conv_W"+str(filter_size)] = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_W"+str(filter_size))
            weights["conv_b"+str(filter_size)] = tf.Variable(tf.constant(0.1, shape=[self.n_filters]), name="conv_b"+str(filter_size))
        return weights

    def conv(self, weights, is_training):
        '''Create a convolution + maxpool layer for each filter size'''
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            W = weights["conv_W"+str(filter_size)]
            b = weights["conv_b"+str(filter_size)]
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv_ = tf.nn.conv2d(
                    self.emb_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.leaky_relu(tf.nn.bias_add(conv_, b), name="relu")
                # h = layers.batch_norm(h, updates_collections=None,
                #                          decay=0.99,
                #                          scale=True, center=True,
                #                          is_training=is_training)
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
        with tf.variable_scope('pretrain_model', reuse=None) as training_scope:
            weights = {}
            if self.freeze_opt == None:
                weights = self.build_emb_weights(weights)
                weights = self.build_conv_weights(weights)
                weights = self.build_fc_weights(self.n_filters * len(self.filter_sizes), weights)

                with tf.variable_scope("embedding"):
                    self.embedding(x, weights["emb_W"], weights["emb_mask_W"])

                # convolutional network
                with tf.variable_scope("conv"):
                    hout = self.conv(weights, is_training)

                with tf.variable_scope("dropout"):
                    h_ = layers.dropout(hout, keep_prob=dropout)

                for i, dim in enumerate(self.dim_hidden[:-1]):
                    h_ = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)])
                    h_ = tf.nn.dropout(h_, dropout)

                # Logits linear layer, i.e. softmax without normalization.
                N, Min = h_.get_shape()
                i = len(self.dim_hidden)-1
                logits = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)], relu=False)

            else:
                with tf.variable_scope("embedding"):
                    Wemb = self.finetune_weights["emb_W"]
                    Wemb_mask = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)
                    self.embedding(x, Wemb, Wemb_mask)

                # convolutional network
                with tf.variable_scope("conv"):
                    # w = {}
                    # for i, filter_size in enumerate(self.filter_sizes):
                    #     w["conv_W"+str(filter_size)] = self.finetune_weights["conv_W"+str(filter_size)]
                    #     w["conv_b"+str(filter_size)] = self.finetune_weights["conv_b"+str(filter_size)]
                    hout = self.conv(self.finetune_weights, is_training)

                with tf.variable_scope("dropout"):
                    h_ = layers.dropout(hout, keep_prob=dropout)

                for i, dim in enumerate(self.dim_hidden[:-1]):
                    Wfc = self.finetune_weights["fc_W"+str(i)]
                    bfc = self.finetune_weights["fc_b"+str(i)]
                    h_ = self.fc(h_, Wfc, bfc)
                    h_ = tf.nn.dropout(h_, dropout)

                i = len(self.dim_hidden)-1
                weights = {}
                dim_in = self.n_hidden_2
                weights["fc_W"+str(i)] = self.weight_variable([int(dim_in), FLAGS.n_classes], name="fc_W"+str(i))
                weights["fc_b"+str(i)] = self.bias_variable([FLAGS.n_classes], name="fc_b"+str(i))

                # Logits linear layer, i.e. softmax without normalization.
                N, Min = h_.get_shape()
                i = len(self.dim_hidden)-1
                logits = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)], relu=False)



                        # elif self.freeze_opt == "mlp":
                        #     weights = self.build_emb_weights(weights)
                        #     weights = self.build_conv_weights(weights)
                        #     for i, dim in enumerate(self.dim_hidden):
                        #         dim_out = dim
                        #         weights["fc_W"+str(i)] = self.finetune_weights["fc_W"+str(i)]
                        #         weights["fc_b"+str(i)] = self.finetune_weights["fc_b"+str(i)]
                        #         dim_in = dim_out
                        # elif self.freeze_opt == "emb":
                        #     weights["emb_W"] = self.finetune_weights["emb_W"]
                        #     weights["emb_mask_W"] = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)
                        #     weights = self.build_conv_weights(weights)
                        #     weights = self.build_fc_weights(self.n_filters * len(self.filter_sizes), weights)
                        # elif self.freeze_opt == "cnn":
                        #     weights = self.build_emb_weights(weights)
                        #     weights = self.build_fc_weights(self.n_filters * len(self.filter_sizes), weights)
                        #     for i, filter_size in enumerate(self.filter_sizes):
                        #         weights["conv_W"+str(filter_size)] = self.finetune_weights["conv_W"+str(filter_size)]
                        #         weights["conv_b"+str(filter_size)] = self.finetune_weights["conv_b"+str(filter_size)]
                        # elif self.freeze_opt == "cnn+emb":


        return logits
