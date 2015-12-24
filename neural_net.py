import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Implement shallow 3-layer network"""

    # alpha - step size of gradient descent optimizer
    # epochs - number of passes through dataset
    # dropout - probability of turning off neurons (set to 0 in predict)
    def __init__(self, alpha=0.01, batch_size=50, num_epochs=10, dropout=0.5, hidden_size=200):
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.classes_ = None
        self.Wh = None
        self.bh = None
        self.Wo = None
        self.bo = None
        pass

    def fit(self, x, y):
        x, y = check_X_y(x, y, 'csr')
        _, n_features = x.shape

        label_bin = LabelBinarizer()
        y = label_bin.fit_transform(y)
        self.classes_ = label_bin.classes_

        y = y.astype(np.float64)

        self._fit_tf(x, y)

    def _fit_tf(self, X, Y):
        # Load and convert data
        n_features = X.shape
        print X.shape
        n_examples = X.shape[1]
        n_classes = Y.shape[1]

        # Input data
        input_features = tf.constant(X.todense(), dtype=tf.float32)
        input_labels = tf.constant(Y, dtype=tf.float32)
        features, label = tf.train.slice_input_producer(
                [input_features, input_labels],
                num_epochs=self.num_epochs
        )
        label = tf.cast(label, tf.int32)
        Xs, ys = tf.train.batch(
                [features, label],
                batch_size=self.batch_size
        )

        # Set up model
        # Weight and bias to hidden layer
        wh = tf.Variable(tf.zeros([n_features, self.hidden_size]))
        bh = tf.Variable(tf.zeros([self.hidden_size]))
        h = tf.nn.relu(tf.matmul(Xs, wh) + bh)

        # Weight and bias to logit output (pre softmax)
        wy = tf.Variable(tf.zeros([self.hidden_size, n_classes]))
        by = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(h, wy) + by

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, ys)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdagradOptimizer(self.alpha)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step)

        # Init for run
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Run iterations
        step = 0
        try:
            while not coord.should_stop():
                # Run one step of the model.
                _, loss_value = sess.run([train_op, loss])
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (self.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)

        self.Wh = wh.eval(sess)
        self.bh = bh.eval(sess)
        self.Wo = wy.eval(sess)
        self.bo = by.eval(sess)
        sess.close()

    def predict(self, x):
        return max(self.predict_proba(x))

    def predict_proba(self, x):
        h = np.max(np.zeros(x.shape()), np.add(np.dot(x, self.Wh), self.bh))
        y = softmax(np.add(np.dot(h, self.Wo), self.bo))
        return y


def softmax(w):
    w = np.array(w)

    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1)

    return dist
