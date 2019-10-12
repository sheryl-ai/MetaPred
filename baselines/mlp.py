import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import os, time, shutil, collections

class MLP(object):
    """
    Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron).
    """
    def __init__(self, num_input, num_classes):
        # Training Parameters
        self.learning_rate = 0.1
        self.batch_size = 64
        self.num_epochs = 200
        self.display_step = 10000
        self.dropout = 0.8
        self.decay_rate = 0.9
        self.decay_steps = 5000/ self.batch_size
        self.momentum = 0.95
        self.patience = 5
        self.eval_frequency = self.num_epochs
        self.regularization = 0.01
        self.regularizers = []
        self.isReg = True
        self.dir_name =  "mlp"

        # Network Parameters
        self.n_hidden_1 = 128 # 1st layer number of neurons
        self.n_hidden_2 = 128 # 2nd layer number of neurons
        self.num_input = num_input
        self.num_classes = num_classes
        self.M = [self.n_hidden_1, self.n_hidden_2, self.num_classes]

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
            op_logits = self.inference(self.ph_data, self.ph_dropout)
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
            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end, :]

            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}

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

    def weight_variable(self, shape):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if self.isReg:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def bias_variable(self, shape):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if self.isReg:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self.weight_variable([int(Min), Mout])
        b = self.bias_variable([Mout])
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    # Create model
    def inference(self, x, dropout):
        for i, dim in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, dim)
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            prob = self.fc(x, self.M[-1], relu=False)
        return prob


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
            batch_data, batch_labels = X_tr[idx, :], y_tr[idx]

            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
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
        writer.close()
        sess.close()
        t_step = (time.time() - t_wall) / num_steps

        return  aucs, accuracies, losses
