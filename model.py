""" Code for the MetaPred algorithm and network architecture. """
import numpy as np
import sklearn
import tensorflow as tf
import os, time, shutil, collections

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

    def convert_to_array(self, data):
        '''convert other type to numpy array'''
        if type(data) is not np.ndarray:
            # data = np.array(data)
            data = data.toarray()  # convert sparse matrices
        return data

     # Helper methods.
    def _get_path(self, folder):
        path = '../../models/'
        return os.path.join(path, folder, self.dir_name)

    def _get_session(self, sess=None):
        '''Restore parameters if no session given.'''
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _get_prediction(self, logits):
        '''Return the predicted classes.'''
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
        return prediction

    def loss_func(self, pred, label):
        '''cross entropy'''
        # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
        label = tf.one_hot(label, FLAGS.n_classes)
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label) / FLAGS.update_batch_size


class MetaPred(BaseModel):
    def __init__(self, data_loader, meta_lr=1e-3, update_lr=1e-2, test_num_updates=-1):
        """
        Args:
            dim_input: dimension of input data (for mlps)
            n_tasks: task number including both source and target
            meta_lr: the base learning rate of the generator
            update_lr: step size alpha for inner gradient update
        """
        super().__init__()

        self.data_loader = data_loader
        self.dim_input = data_loader.dim_input
        self.n_tasks = data_loader.n_tasks
        self.meta_lr = meta_lr
        self.update_lr = update_lr
        self.test_num_updates = test_num_updates
        self.auc_stable = []
        self.f1s_stable = []

        self.weights_for_finetune = dict() # to store the value of learned params

        print('method:', "meta-"+FLAGS.method, 'data shape:', self.dim_input, 'meta-bz:', FLAGS.meta_batch_size, 'update-bz:', FLAGS.update_batch_size, \
             'num update:', FLAGS.num_updates, 'meta-lr:', meta_lr, 'update-lr:', update_lr)

        if FLAGS.method == "cnn":
            # sequential network (cnn) configuration
            self.cnn_config(data_loader)
        elif FLAGS.method == "rnn":
            # sequential network (cnn) configuration
            self.rnn_config(data_loader)

        # Build the computational graph.
        self.build_graph()

    ####################################### Networks #######################################
    def weight_variable(self, shape, name='weights'):
        if FLAGS.pretrain:
            initial = self.pretrain_weights[name]
            var = tf.Variable(initial_value=initial, name=name)
        else:
            initial = tf.truncated_normal_initializer(0, 0.1)
            var = tf.get_variable(name, shape, tf.float32, initializer=initial)

        if FLAGS.isReg:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def bias_variable(self, shape, initial=None, name='bias'):
        if FLAGS.pretrain:
            initial = self.pretrain_weights[name]
            var = tf.Variable(initial_value=initial, name=name)
        else:
            initial = tf.constant_initializer(0.1)
            var = tf.get_variable(name, shape, tf.float32, initializer=initial)

        if FLAGS.isReg:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    ############################### Fully Conneted Network #################################
    # construct weights
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

    ############################ Embedding Layer for SeqNet ################################
    def build_emb_weights(self, weights):
        weights["emb_W"] = tf.Variable(tf.random_normal([self.n_words, self.n_hidden], stddev=self.init_std), name="emb_W")
        with tf.variable_scope("emb", reuse=tf.AUTO_REUSE) as scope:
            weights["emb_mask_W"] = tf.get_variable("mask_padding", initializer=MASK_ARRAY, dtype="float32", trainable=False)
        return weights

    def embedding(self, x, Wemb, Wemb_mask):
        _x = tf.nn.embedding_lookup(Wemb, x) # recs size is (batch_size, timesteps, code_size)
        _x_mask = tf.nn.embedding_lookup(Wemb_mask, x)
        # print (_x.get_shape())
        # print (_x_mask.get_shape())
        emb_vecs = tf.multiply(_x, _x_mask)
        emb_vecs = tf.reduce_sum(emb_vecs, 2)
        # print (emb_vecs.get_shape())
        return emb_vecs

    ############################ Convolutional Neural Network ##############################
    def cnn_config(self, data_loader, init_std=0.05):
        # Network Parameters
        self.init_std = init_std
        self.n_hidden = 256 # hidden dimensions of embedding
        self.n_hidden_1 = 128
        self.n_hidden_2 = 128
        self.n_words = data_loader.n_words
        self.n_classes = FLAGS.n_classes
        self.n_filters = 128
        self.num_input =  data_loader.dim_input
        self.timesteps = data_loader.timesteps
        self.code_size = data_loader.code_size
        self.dim_hidden = [self.n_hidden_1, self.n_hidden_2, FLAGS.n_classes] # for AD
        self.filter_sizes = [3, 4, 5]
        self.learner = self.cnn_sequential

    def build_conv_weights(self, weights):
        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.n_hidden, 1, self.n_filters]
            weights["conv_W"+str(filter_size)] = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_W"+str(filter_size))
            weights["conv_b"+str(filter_size)] = tf.Variable(tf.constant(0.1, shape=[self.n_filters]), name="conv_b"+str(filter_size))
        return weights

    def conv(self, emb_vecs, weights, is_training=True):
        '''Create a convolution + maxpool layer for each filter size'''
        pooled_outputs = []
        emb_expanded = tf.expand_dims(emb_vecs, -1)
        # print(emb_expanded.get_shape())
        for i, filter_size in enumerate(self.filter_sizes):
            W = weights["conv_W"+str(filter_size)]
            b = weights["conv_b"+str(filter_size)]
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv_ = tf.nn.conv2d(
                    emb_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.leaky_relu(tf.nn.bias_add(conv_, b), name="relu")
                with tf.name_scope("bnorm{}".format(filter_size)) as scope:
                    h = layers.batch_norm(h, updates_collections=None,
                                             decay=0.99,
                                             scale=True, center=True,
                                             is_training=is_training, reuse=tf.AUTO_REUSE, scope=scope)
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

    def cnn_sequential(self, x, weights, dropout, reuse=False, is_training=True, type="source"):
        xemb = self.embedding(x, weights["emb_W"], weights["emb_mask_W"])

        # convolutional network
        hout = self.conv(xemb, weights, is_training)

        h_ = layers.dropout(hout, keep_prob=dropout)

        for i, dim in enumerate(self.dim_hidden[:-1]):
            h_ = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)])
            h_ = tf.nn.dropout(h_, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        N, Min = h_.get_shape()
        i = len(self.dim_hidden)-1
        logits = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)], relu=False)
        return logits

    ############################ Recurrent Neural Network ##############################
    def rnn_config(self, data_loader, init_std=0.05):
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
        self.learner = self.rnn_sequential

    def build_lstm_weights(self, weights):
        # # Keep W_xh and W_hh separate here as well to reuse initialization methods
        # with tf.variable_scope(scope or type(self).__name__):
        weights["lstm_W_xh"] = tf.get_variable('lstm_W_xh', [self.n_hidden, 4 * self.n_hidden],
                               initializer=self.orthogonal_initializer())
        weights["lstm_W_hh"] = tf.get_variable('lstm_W_hh', [self.n_hidden, 4 * self.n_hidden],
                               initializer=self.lstm_identity_initializer(0.95),)
        weights["lstm_b"] = tf.get_variable('lstm_b', [4 * self.n_hidden])
        return weights

    def lstm_identity_initializer(self, scale):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            """Ugly cause LSTM params calculated in one matrix multiply"""
            size = shape[0]
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

    def rnn_sequential(self, x, weights, dropout, reuse=False, is_training=True, type='source'):
        # embedding
        xemb = self.embedding(x, weights["emb_W"], weights["emb_mask_W"])

        # recurrent neural networks
        xemb = tf.unstack(xemb, self.timesteps, 1)
        lstm_cell = LSTMCell(self.n_hidden, weights["lstm_W_xh"], weights["lstm_W_hh"], weights["lstm_b"])
        #c, h
        if type == "source":
            W_state_c = tf.random_normal([(self.n_tasks-1)*FLAGS.update_batch_size, self.n_hidden], stddev=0.1)
            W_state_h = tf.random_normal([(self.n_tasks-1)*FLAGS.update_batch_size, self.n_hidden], stddev=0.1)
        elif type == "target":
            W_state_c = tf.random_normal([FLAGS.update_batch_size, self.n_hidden], stddev=0.1)
            W_state_h = tf.random_normal([FLAGS.update_batch_size, self.n_hidden], stddev=0.1)
        # outputs, state = tf.nn.dynamic_rnn(lstm_cell, xemb, initial_state=(W_state_c, W_state_h), dtype=tf.float32)
        outputs, state = tf.nn.static_rnn(lstm_cell, xemb, initial_state=(W_state_c, W_state_h), dtype=tf.float32)
        _, hout = state

        with tf.variable_scope("dropout"):
            h_ = layers.dropout(hout, keep_prob=dropout)

        for i, dim in enumerate(self.dim_hidden[:-1]):
            h_ = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)])
            h_ = tf.nn.dropout(h_, dropout)

        x_rep = tf.identity(h_)

        # Logits linear layer, i.e. softmax without normalization.
        N, Min = h_.get_shape()
        i = len(self.dim_hidden)-1
        logits = self.fc(h_, weights["fc_W"+str(i)], weights["fc_b"+str(i)], relu=False)
        return logits, x_rep


    def build_graph(self):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                self.input_s = tf.placeholder(tf.int32, (FLAGS.meta_batch_size, (self.n_tasks-1) * FLAGS.update_batch_size, self.timesteps, self.code_size), 'source_x')
                self.input_t = tf.placeholder(tf.int32, (FLAGS.meta_batch_size, FLAGS.update_batch_size, self.timesteps, self.code_size), 'target_x')
                self.label_s = tf.placeholder(tf.int64, (FLAGS.meta_batch_size, (self.n_tasks-1) * FLAGS.update_batch_size), 'source_y')
                self.label_t = tf.placeholder(tf.int64, (FLAGS.meta_batch_size, FLAGS.update_batch_size), 'target_y')

                self.ph_training = tf.placeholder(tf.bool, name='trainingFlag')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            # construct metatrain_ and metaval_
            if FLAGS.method == "cnn" or FLAGS.method == "rnn":
                self.build_model((self.input_s, self.input_t, self.label_s, self.label_t), prefix='metatrain_', is_training=self.ph_training)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            self.op_weights = self.get_op_variables()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
            print ('graph built!')

        self.graph.finalize()

    def get_op_variables(self):
        if FLAGS.method == "cnn":
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
        elif FLAGS.method == "rnn":
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
        return op_weights

    def build_weights(self):
        weights = {}
        if FLAGS.method == "cnn":
            weights = self.build_emb_weights(weights)
            weights = self.build_conv_weights(weights)
            weights = self.build_fc_weights(self.n_filters * len(self.filter_sizes), weights)
        elif FLAGS.method == "rnn":
            weights = self.build_emb_weights(weights)
            weights = self.build_lstm_weights(weights)
            weights = self.build_fc_weights(self.n_hidden, weights)
        return weights


    def build_model(self, input_tensors, prefix='metatrain_', is_training=True):
        """
        Args:
            input_tensors = []:
                source_xb:   [batch_size, (n_tasks-1)*update_batch_size, data_shape]
                source_yb:   [batch_size, (n_tasks-1)*update_batch_size, ]
                target_xb:   [batch_size, update_batch_size, data_shape]
                target_yb:   [batch_size, update_batch_size, ] i.e., querysz = 1
            # update_batch_size: number of examples used for inner gradient update (K for K tasks)
            # meta_batch_size: number of mate-batches sampled per meta-update
            prefix:        pretrain_/metatrain_/metaval_/metatest_, for training, we build train val and test network meanwhile.
        """
        # source: training data for inner gradient, target: test data for meta gradient
        source_xb, target_xb, source_yb, target_yb = input_tensors

        # create or reuse network variable, not including batch_norm variable, therefore we need extra reuse mechnism
        # to reuse batch_norm variables.
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
            # Define the weights. weights is a dictionary
            self.weights = weights = self.build_weights()

            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            # target_preds_tasks[i] and target_losses_tasks[i] is the output and loss after i+1 gradient updates
            source_pred_tasks, source_loss_tasks, source_acc_tasks, source_auc_tasks = [], [], [], [] # source and target has seperate loss
                                                                                                      # and accuracies
            target_losses_tasks = [[]]*num_updates # result of every updates for test data
            target_preds_tasks = [[]]*num_updates # prediction
            target_accs_tasks = [[]]*num_updates
            target_aucs_tasks = [[]]*num_updates

            def task_metalearn(input, reuse=True):
                """
                Perform gradient descent for one task in the meta-batch.
                Args:
                    source_x:   [(n_tasks-1)*update_batch_size, data_shape]
                    source_y:   [(n_tasks-1)*update_batch_size, ]
                    target_x:   [update_batch_size, data_shape]
                    target_y:   [update_batch_size, ]
                    training:   training or not, for batch_norm
                """
                source_x, target_x, source_y, target_y = input # map_fn only support one parameters, so we need to unpack from tuple
                # print (source_x.get_shape())
                # print (target_x.get_shape())
                # print (source_y.get_shape())
                # print (target_y.get_shape())

                # record the op in t update step, each element is the results of the upate step.
                target_preds, target_losses, target_accs, target_aucs, target_represents = [], [], [], [], []

                # That's, to create variable, you must turn off reuse
                source_pred, _ = self.learner(source_x, weights, self.ph_dropout, reuse=False, is_training=is_training, type="source")
                # print (source_pred.get_shape())
                source_loss = self.loss_func(source_pred, source_y)
                source_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(source_pred), 1), source_y)

                # compute gradients
                grads = tf.gradients(source_loss, list(weights.values()))

                if FLAGS.stop_grad: # if True, do not use second derivatives in meta-optimization (for speed)
                    grads = [tf.stop_gradient(grad) for grad in grads]

                # grad and variable dict
                gvs = dict(zip(weights.keys(), grads))
                # theta_pi = theta - alpha * grads
                fast_weights = dict(zip(weights.keys(), [weights[key] - tf.multiply(self.update_lr, gvs[key]) for key in weights.keys()]))
                # fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gvs[key] for key in weights.keys()]))

                # use theta_pi for fast adaption
                target_pred, target_represent = self.learner(target_x, fast_weights, self.ph_dropout, reuse=True, is_training=is_training, type="target")
                target_loss = self.loss_func(target_pred, target_y)
                target_preds.append(target_pred)
                target_losses.append(target_loss)
                target_represents.append(target_represent)

                # continue to build T1-TK steps graph
                for _ in range(1, num_updates): # i.e., num_updates = 4, update 3 times
                    # T_k loss on meta-train
                    # we need meta-train loss to fine-tune the task and meta-test loss to update theta
                    loss = self.loss_func(self.learner(source_x, fast_weights, self.ph_dropout, reuse=True, is_training=is_training, type="source")[0], source_y)
                    # compute gradients
                    grads = tf.gradients(loss, list(fast_weights.values()))

                    # compose grad and variable dict
                    gvs = dict(zip(fast_weights.keys(), grads))
                    # update theta_pi according to varibles
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - tf.multiply(self.update_lr, gvs[key])
                                          for key in fast_weights.keys()]))

                    # forward on theta_pi
                    target_pred, target_represent = self.learner(target_x, fast_weights, self.ph_dropout, reuse=True, is_training=is_training, type="target")
                    # we need accumulate all meta-test losses to update theta
                    target_loss = self.loss_func(target_pred, target_y)
                    target_preds.append(target_pred)
                    target_losses.append(target_loss)
                    target_represents.append(target_represent)


                task_output = [target_represents, source_pred, target_preds, source_loss, target_losses]
                for j in range(num_updates):
                    target_accs.append(tf.contrib.metrics.accuracy(predictions=tf.argmax(tf.nn.softmax(target_preds[j]), 1), labels=target_y))
                task_output.extend([source_acc, target_accs])
                return task_output

            if FLAGS.norm is not 'None': # batch norm or layer norm
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((source_xb[0], target_xb[0], source_yb[0], target_yb[0]), False)

            out_dtype = [[tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates,
                         tf.float32, [tf.float32] * num_updates]

            result = tf.map_fn(task_metalearn, elems=(source_xb, target_xb, source_yb, target_yb),
                              dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size, name='map_fn')
            target_represents_tasks, source_pred_tasks, target_preds_tasks, source_loss_tasks, target_losses_tasks, \
                           source_acc_tasks, target_accs_tasks = result

        ## Performance & Optimization
        # average loss
        self.source_loss = source_loss = tf.reduce_sum(source_loss_tasks) / FLAGS.meta_batch_size
        # [avgloss_T1, avgloss_T2, ..., avgloss_TK]
        self.target_losses = target_losses = [tf.reduce_sum(target_losses_tasks[j]) / FLAGS.meta_batch_size
                                              for j in range(num_updates)]
        self.source_acc = source_acc = tf.reduce_sum(source_acc_tasks) / FLAGS.meta_batch_size
        self.target_accs = target_accs = [tf.reduce_sum(target_accs_tasks[j]) / FLAGS.meta_batch_size
                                            for j in range(num_updates)]
        self.source_pred = source_pred_tasks
        self.target_preds = target_preds_tasks[FLAGS.num_updates-1]
        self.target_represent = target_represents_tasks[FLAGS.num_updates-1]

        if self.ph_training is not False:
            # meta-train optim
            optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
            # meta-train gradients, target_losses[-1] is the accumulated loss across over tasks.
            self.gvs = gvs = optimizer.compute_gradients(self.source_loss + self.target_losses[FLAGS.num_updates-1])
            # update theta
            self.metatrain_op = optimizer.apply_gradients(gvs)

        ## Summaries
        # NOTICE: every time build model, support_loss will be added to the summary, but it's different.
        tf.summary.scalar(prefix+'Pre-update loss', source_loss)
        tf.summary.scalar(prefix+'Pre-update accuracy', source_acc)
        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), target_losses[j])
            tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), target_losses[j])


    def compute_metrics(self, predictions, labels):
        '''compute metrics score'''
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
        auc = sklearn.metrics.auc(fpr, tpr)
        ncorrects = sum(predictions == labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        ap = sklearn.metrics.average_precision_score(labels, predictions, 'micro')
        f1score = sklearn.metrics.f1_score(labels, predictions,  'micro')
        return auc, ap, f1score


    # def evaluate(self, sample, label, sess=None, prefix="metaval_"):
    def evaluate(self, episode, data_tuple_val, sess=None, prefix="metaval_"):
        '''validate meta learning model'''
        target_acc,target_vals,target_preds = [], [], []
        size = len(episode)

        for begin in range(0, size, FLAGS.meta_batch_size):
            end = begin + FLAGS.meta_batch_size
            end = min([end, size])
            if end-begin < FLAGS.meta_batch_size: break

            batch_idx = range(begin, end)
            sample, label = self.get_feed_data(episode, batch_idx, data_tuple_val, is_training=False)

            X_tensor_s = self.convert_to_array(sample[:, :(self.n_tasks-1) * FLAGS.update_batch_size, :, :])
            X_tensor_t = self.convert_to_array(sample[:, (self.n_tasks-1) * FLAGS.update_batch_size:, :, :])
            y_tensor_s = self.convert_to_array(label[:, :(self.n_tasks-1) * FLAGS.update_batch_size])
            y_tensor_t = self.convert_to_array(label[:, (self.n_tasks-1) * FLAGS.update_batch_size:])

            feed_dict = {self.input_s: X_tensor_s, self.input_t: X_tensor_t, self.label_s: y_tensor_s, self.label_t: y_tensor_t, self.ph_dropout: 1, self.ph_training: False}
            input_tensors = [self.target_preds, self.target_accs[FLAGS.num_updates-1]]
            metaval_target_preds, metaval_target_accs = sess.run(input_tensors, feed_dict)
            target_acc.append(metaval_target_accs)
            target_preds.append(metaval_target_preds)
            target_vals.append(y_tensor_t)

        target_vals = np.array(target_vals).flatten()
        target_preds = np.array([np.argmax(preds, axis=2) for preds in target_preds]).flatten()

        target_acc = np.mean(target_acc)
        target_auc, target_ap, target_f1 = self.compute_metrics(target_preds, target_vals)

        return target_acc, target_auc, target_ap, target_f1


    def get_feed_data(self, episode, batch_idx, data_tuple, is_training, is_show=False):
        ''' given batch indices, get data array from the generated index episodes'''
        n_samples_per_task = FLAGS.update_batch_size
        data_s, data_t, label_s, label_t  = data_tuple
        # generate episode
        sample, label = [], []
        batch_count = 0
        for i in range(len(batch_idx)): # the 1st dimension is the batch size
            # i.e., sample 16 patients from selected tasks
            # len of spl and lbl: 4 * 16
            spl, lbl = [], [] # samples and labels in one episode
            bi = batch_idx[i]
            data_idx = episode[bi] # all tasks are merged: [task1, task2, ..., tastn], where taskn is target
            n_source = 0
            for i in range(len(self.data_loader.source)):
                s_idx = data_idx[i*n_samples_per_task:(i+1)*n_samples_per_task]
                spl.extend(data_s[i][s_idx])
                lbl.extend(label_s[i][s_idx])
                n_source += n_samples_per_task
            ### do not keep pos/neg ratio
            if is_training:
                t_idx = data_idx[n_source:]
                spl.extend(data_t[0][t_idx])
                lbl.extend(label_t[0][t_idx])
            else:
                t_idx = data_idx[n_source:]
                spl.extend(data_t[t_idx])
                lbl.extend(label_t[t_idx])

            batch_count += 1
            # add meta_batch
            sample.append(spl)
            label.append(lbl)

        sample = np.array(sample, dtype="float32")
        label = np.array(label, dtype="float32")
        return sample, label


    def fit(self, episode, episode_val, ifold, exp_string, model_file = None):
        sess = tf.Session(graph=self.graph)
        if FLAGS.resume or not FLAGS.train:
            model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
            if model_file:
                ind1 = model_file.index('model')
                print("Restoring model weights from " + model_file)
                self.op_saver.restore(sess, model_file)
        sess.run(self.op_init)

        if FLAGS.log:
            train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)

        # load data for metatrain
        data_tuple = (self.data_loader.data_s, self.data_loader.data_t, self.data_loader.label_s, self.data_loader.label_t)
        # load data for metaeval
        data_tuple_val = (self.data_loader.data_s, self.data_loader.data_tt_val[ifold], self.data_loader.label_s, self.data_loader.label_tt_val[ifold])

        prelosses, postlosses, preaccs, postaccs = [], [], [], []

        # train for meta_iteartion epoches
        indices = collections.deque()
        for itr in range(FLAGS.metatrain_iterations):
            feed_dict = {}
            input_tensors = [self.metatrain_op]

            if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
                input_tensors.extend([self.op_summary, self.source_loss, self.target_losses[FLAGS.num_updates-1]])
                input_tensors.extend([self.source_acc, self.target_accs[FLAGS.num_updates-1], self.target_preds])

            if len(indices) < FLAGS.meta_batch_size:
                 indices.extend(np.random.permutation(len(episode)))
            batch_idx = [indices.popleft() for i in range(FLAGS.meta_batch_size)]
            sample, label = self.get_feed_data(episode, batch_idx, data_tuple, is_training=True)

            X_tensor_s = self.convert_to_array(sample[:, :(self.n_tasks-1) * FLAGS.update_batch_size, :, :])
            X_tensor_t = self.convert_to_array(sample[:, (self.n_tasks-1) * FLAGS.update_batch_size:, :, :])
            y_tensor_s = self.convert_to_array(label[:, :(self.n_tasks-1) * FLAGS.update_batch_size])
            y_tensor_t = self.convert_to_array(label[:, (self.n_tasks-1) * FLAGS.update_batch_size:])
            feed_dict = {self.input_s: X_tensor_s, self.input_t: X_tensor_t, self.label_s: y_tensor_s, self.label_t: y_tensor_t, self.ph_dropout: FLAGS.dropout, self.ph_training: True}

            result = sess.run(input_tensors, feed_dict)
            if itr % SUMMARY_INTERVAL == 0:
                prelosses.append(result[-5])
                preaccs.append(result[-3])
                if FLAGS.log:
                    train_writer.add_summary(result[1], itr)
                postlosses.append(result[-4])
                postaccs.append(result[-2])
                postauc, postap, postf1 = self.compute_metrics(np.argmax(result[-1], axis=2).flatten(), y_tensor_t.flatten())

            if (itr!=0) and itr % PRINT_INTERVAL == 0:
                print_str = 'Iteration ' + str(itr)
                print_str += ': sacc: ' + str(np.mean(preaccs)) + ', tacc: ' + str(np.mean(postaccs))
                print_str += " tauc: " + str(postauc) + " tap: " + str(postap) + " tf1: " + str(postf1)
                print(print_str)
                preaccs, postaccs = [], []
                prelosses, postlosses = [], []

            if (itr!=0) and itr % SAVE_INTERVAL == 0:
                self.op_saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

            if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
                target_accs, target_aucs, target_ap, target_f1s = self.evaluate(episode_val, data_tuple_val, sess=sess, prefix="metaval_")
                self.auc_stable.append(target_aucs)
                self.f1s_stable.append(target_f1s)
                print('Validation results: ' + "tAcc: " + str(target_accs) + ", tAuc: " + str(target_aucs) + ", tAP: "  + str(target_ap) + ", tF1: "  + str(target_f1s))
                print ("---------------")
        self.op_saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
        print ("---------------")

        # store weights value for fine-tune
        feed_dict = {}
        for k in self.op_weights:
             self.weights_for_finetune[k] = sess.run([self.op_weights[k]], feed_dict)[0]
        return sess


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
