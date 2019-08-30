import sklearn.gaussian_process as gp
import sklearn.neural_network as nn
import sklearn.preprocessing as prepro
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
from scipy import *
import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from statsmodels.tsa.arima_model import ARIMA
import warnings

## Set inputs parameters
delays = 10
percent_for_test = 0.2
#features = []

## Auto Regressive Integrated Moving Average using statsmodel
# http://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html
AR = 2
I = 1
MA = 0

## Create Multi Layer Perceptron Neural Network from sci kit learn library
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
mlp = nn.MLPRegressor(hidden_layer_sizes=(400,200), # number of neurons on each hidden layer
                      activation='relu',            # activation function {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
                      solver='adam',                # solver to minimize loss function {‘lbfgs’, ‘sgd’, ‘adam’}
                      alpha=0.01,                   # penalty parameter
                      learning_rate='constant',     # function for learning rate {'constant', ‘invscaling’, ‘adaptive’}
                      learning_rate_init=0.01,      # learning rate first value
                      power_t=0.5,                  # the exponent for inverse scaling learning rate. It is used in updating effective learning rate
                      max_iter=10000,               # maximum number of iterations
                      shuffle=True,                 # whether to shuffle samples in each iteration
                      random_state=None,            # if int, random_state is the seed used by the random number generator
                      tol=0.0000001,                 # tolerance for optimization in n_iter_no_change iterations
                      verbose=True,                 # print loss every iteration
                      validation_fraction=0.15,      # proportion of set to use as validation
                      n_iter_no_change=1000 )       # iterations to look at tol


## Create Gaussian Process Regressor using a Radial Basis Function kernel from sci kit learn library
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
rbf = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(1.0),  # kernel function https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
                                  alpha=0.01,                  # value added to the diagonal of the kernel matrix during fitting
                                  optimizer='fmin_l_bfgs_b',   # function to optimize kernel’s parameters minimizing loss
                                  n_restarts_optimizer= 50,   # numbers to reoptimize
                                  random_state=1)              # if int, random_state is the seed used by the random number generator


## Adaptative Neuro Fuzzy Interference System using Tensor Flow library
# https://github.com/tiagoCuervo/TensorANFIS

anfis_rules = 100          # number of rules
anfis_lr = 0.01            # learning rate
anfis_num_epochs = 1000   # epochs
anfis_verbose = True       # print loss every iteration

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class ANFIS:

    def __init__(self, n_inputs, n_rules, learning_rate=1e-2):
        self.n = n_inputs
        self.m = n_rules
        self.inputs = tf.placeholder(tf.float32, shape=(None, n_inputs))  # Input
        self.targets = tf.placeholder(tf.float32, shape=None)  # Desired output
        mu = tf.get_variable("mu", [n_rules * n_inputs],
                             initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        sigma = tf.get_variable("sigma", [n_rules * n_inputs],
                                initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS
        y = tf.get_variable("y", [1, n_rules], initializer=tf.random_normal_initializer(0, 1))  # Sequent centers

        self.params = tf.trainable_variables()

        self.rul = tf.reduce_prod(
            tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), mu)) / tf.square(sigma)),
                       (-1, n_rules, n_inputs)), axis=2)  # Rule activations
        # Fuzzy base expansion function:
        num = tf.reduce_sum(tf.multiply(self.rul, y), axis=1)
        den = tf.clip_by_value(tf.reduce_sum(self.rul, axis=1), 1e-12, 1e12)
        self.out = tf.divide(num, den)

        self.loss = tf.losses.huber_loss(self.targets, self.out)  # Loss function computation
        # Other loss functions for regression, uncomment to try them:
        # loss = tf.sqrt(tf.losses.mean_squared_error(target, out))
        # loss = tf.losses.absolute_difference(target, out)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)  # Optimization step
        # Other optimizers, uncomment to try them:
        # self.optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)
        # self.optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.init_variables = tf.global_variables_initializer()  # Variable initializer

    def infer(self, sess, x, targets=None):
        if targets is None:
            return sess.run(self.out, feed_dict={self.inputs: x})
        else:
            return sess.run([self.out, self.loss], feed_dict={self.inputs: x, self.targets: targets})

    def train(self, sess, x, targets):
        yp, l, _ = sess.run([self.out, self.loss, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        return l, yp

    def plotmfs(self, sess):
        mus = sess.run(self.params[0])
        mus = np.reshape(mus, (self.m, self.n))
        sigmas = sess.run(self.params[1])
        sigmas = np.reshape(sigmas, (self.m, self.n))
        y = sess.run(self.params[2])
        xn = np.linspace(-1.5, 1.5, 1000)
        for r in range(self.m):
            if r % 4 == 0:
                plt.figure(figsize=(11, 6), dpi=80)
            plt.subplot(2, 2, (r % 4) + 1)
            ax = plt.subplot(2, 2, (r % 4) + 1)
            ax.set_title("Rule %d, sequent center: %f" % ((r + 1), y[0, r]))
            for i in range(self.n):
                plt.plot(xn, np.exp(-0.5 * ((xn - mus[r, i]) ** 2) / (sigmas[r, i] ** 2)))

    def predict(fis, anfis_num_epochs, X_train, y_train, X_test, y_test, anfis_verbose):
        with tf.Session() as sess:
            sess.run(fis.init_variables)
            trn_costs = []
            val_costs = []
            for epoch in range(anfis_num_epochs):
                trn_loss, trn_pred = fis.train(sess, X_train, y_train)
                val_pred, val_loss = fis.infer(sess, X_test, y_test)
                if anfis_verbose:
                    print("Train cost after epoch %i: %f" % (epoch, trn_loss))
                trn_costs.append(trn_loss)
                val_costs.append(val_loss)
        return val_pred