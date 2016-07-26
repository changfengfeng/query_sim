# -*- coding: utf-8 -*-
# @Author: Koth Chen
# @Date:   2016-07-26 13:48:32
# @Last Modified by:   Koth Chen
# @Last Modified time: 2016-07-26 23:35:34
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics

import tensorflow as tf
from tensorflow.contrib import learn
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', "./datas/train.tok",
                           'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', "./datas/test.tok",
                           'Test data dir')
tf.app.flags.DEFINE_string('log_dir', "logs", 'The log  dir')
tf.app.flags.DEFINE_string("word2vec_path", "./datas/vec.txt",
                           "the word2vec data path")

tf.app.flags.DEFINE_integer("max_query_len", 20, "max num of tokens per query")
tf.app.flags.DEFINE_integer("window_size", 5, "window size to do convolution")
tf.app.flags.DEFINE_integer("embedding_size", 100, "embedding size")
tf.app.flags.DEFINE_integer("num_filters", 100, "num of filters")


def do_load_data(path):
  x = []
  y = []
  fp = open(path, "r")
  for line in fp.readlines():
    line = line.rstrip()
    if not line:
      continue
    ss = line.split(" ")
    assert (len(ss) == (FLAGS.max_query_len * 2 + 1))
    y.append(int(ss[FLAGS.max_query_len * 2]))
    lx = []
    for i in range(FLAGS.max_query_len * 2):
      lx.append(int(ss[i]))
    x.append(lx)
  fp.close()
  return np.array(x), np.array(y)


class Model:
  def __init__(self, windowsSize, embeddingSize, numFilters, w2vPath):
    self.windowsSize = windowsSize
    self.embeddingSize = embeddingSize
    self.numFilters = numFilters
    with tf.variable_scope('CNN_Layer') as scope:
      self.filters = tf.get_variable(
          "filters",
          shape=[windowsSize, embeddingSize, 1, numFilters],
          regularizer=tf.contrib.layers.l2_regularizer(0.0001),
          initializer=tf.truncated_normal_initializer(stddev=0.01),
          dtype=tf.float32)
    self.w2v = self.load_w2v(w2vPath)
    self.words = tf.constant(self.w2v, name="words")
    with tf.variable_scope('SIM_MATRIX') as scope:
      self.M = tf.get_variable(
          "sim_maxtrix",
          shape=[numFilters, numFilters],
          regularizer=tf.contrib.layers.l2_regularizer(0.01),
          initializer=tf.truncated_normal_initializer(stddev=0.01),
          dtype=tf.float32)
    with tf.variable_scope('Softmax') as scope:
      self.W = tf.get_variable(
          shape=[numFilters * 2 + 1, 2],
          initializer=tf.truncated_normal_initializer(stddev=0.01),
          name="weights",
          regularizer=tf.contrib.layers.l2_regularizer(0.001))
      self.b = tf.Variable(tf.zeros([2], name="bias"))

    self.inp = tf.placeholder(tf.int32,
                              shape=[None, FLAGS.max_query_len * 2],
                              name="input_placeholder")
    self.tp = tf.placeholder(tf.int32, shape=[None], name="target_placeholder")
    pass

  def load_w2v(self, path):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == (FLAGS.embedding_size))
    ws = []
    mv = [0 for i in range(dim)]
    for t in range(total):
      line = fp.readline().strip()
      ss = line.split(" ")
      assert (len(ss) == (dim + 1))
      vals = []
      for i in range(1, dim + 1):
        fv = float(ss[i])
        mv[i - 1] += fv
        vals.append(fv)
      ws.append(vals)
    for i in range(dim):
      mv[i] = mv[i] / total
    ws.append(mv)
    fp.close()
    return np.asarray(ws, dtype=np.float32)

  def inference(self, X):
    word_vectors = learn.ops.embedding_lookup(self.words, X)
    word_vectors = tf.expand_dims(word_vectors, 3)
    ws = tf.split(1, 2, word_vectors)
    pools = []
    with tf.variable_scope('CNN_Layer') as scope:
      for i in xrange(2):
        # Apply Convolution filtering on input sequence.
        conv1 = tf.nn.conv2d(ws[i],
                             self.filters, [1, 1, 1, 1],
                             padding='VALID')
        conv1 = tf.nn.relu(conv1)
        pool = tf.nn.max_pool(
            conv1,
            ksize=[1, FLAGS.max_query_len - self.windowsSize + 1, 1, 1],
            strides=[1, FLAGS.max_query_len - self.windowsSize + 1, 1, 1],
            padding='SAME')
        pools.append(tf.reshape(pool, [-1, self.numFilters]))
        scope.reuse_variables()
    mv = tf.matmul(pools[0], self.M)
    mv = tf.matmul(mv, tf.transpose(pools[1]))
    simv = tf.diag_part(mv)
    simv = tf.expand_dims(simv, 1)
    features = tf.concat(1, [pools[0], simv, pools[1]])
    with tf.variable_scope('Softmax') as scope:
      return tf.add(tf.matmul(features, self.W), self.b, name="presoftmax")

  def loss(self, X, Y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        self.inference(X), Y))

  def test_correct_count(self):
    logit = self.inference(self.inp)
    logit = tf.nn.softmax(logit)
    predicted = tf.cast(tf.arg_max(logit, 1), tf.int32)
    return tf.reduce_sum(tf.cast(tf.equal(predicted, self.tp), tf.float32))


def read_csv(batch_size, file_name):
  filename_queue = tf.train.string_input_producer(
      [os.path.dirname(__file__) + "/" + file_name])

  reader = tf.TextLineReader(skip_header_lines=0)
  key, value = reader.read(filename_queue)

  # decode_csv will convert a Tensor from type string (the text line) in
  # a tuple of tensor columns with the specified defaults, which also
  # sets the data type for each column
  decoded = tf.decode_csv(
      value,
      field_delim=' ',
      record_defaults=[[0] for i in range(FLAGS.max_query_len * 2 + 1)])

  # batch actually reads the file and loads "batch_size" rows in a single tensor
  return tf.train.shuffle_batch(decoded,
                                batch_size=batch_size,
                                capacity=batch_size * 50,
                                min_after_dequeue=batch_size)


def test_evaluate(sess, tmodel, inp, tp, tX, tY):
  totalEqual = 0
  batchSize = 100
  numBatch = int(tX.shape[0] / batchSize)
  for i in range(numBatch):
    feed_dict = {inp: tX[i * batchSize:(i + 1) * batchSize],
                 tp: tY[i * batchSize:(i + 1) * batchSize]}
    totalEqual += sess.run(tmodel, feed_dict)
  print("accuracy:[%.2f]" % (totalEqual / tX.shape[0]))


def inputs(path):

  whole = read_csv(100, path)
  label = whole[-1]

  # convert class names to a 0 based class index.
  label_number = tf.to_int32(label)

  features = tf.transpose(tf.pack(whole[:-1]))

  return features, label_number


def train(total_loss):
  learning_rate = 0.001
  return tf.train.AdamOptimizer(learning_rate).minimize(total_loss)


def main(unused_argv):
  curdir = os.path.dirname(os.path.realpath(__file__))
  traindir = tf.app.flags.FLAGS.train_data_path
  if not traindir.startswith("/"):
    traindir = curdir + "/" + traindir
  graph = tf.Graph()
  with graph.as_default():
    model = Model(FLAGS.window_size, FLAGS.embedding_size, FLAGS.num_filters,
                  tf.app.flags.FLAGS.word2vec_path)
    print("train path:", traindir)
    X, Y = inputs(traindir)
    tX, tY = do_load_data(tf.app.flags.FLAGS.test_data_path)
    total_loss = model.loss(X, Y)
    train_op = train(total_loss)
    test_correct = model.test_correct_count()
    sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
    with sv.managed_session(master='') as sess:
      # actual training loop
      training_steps = 50000
      for step in range(training_steps):
        if sv.should_stop():
          break
        try:
          sess.run([train_op])
          # for debugging and learning purposes, see how the loss gets decremented thru training steps
          if step % 100 == 0:
            print("[%d] loss: [%r]" % (step, sess.run(total_loss)))
          if step % 500 == 0:
            test_evaluate(sess, test_correct, model.inp, model.tp, tX, tY)
        except KeyboardInterrupt, e:
          sv.saver.save(sess, FLAGS.log_dir + '/model', global_step=step + 1)
          raise e
      sess.close()


if __name__ == '__main__':
  tf.app.run()
