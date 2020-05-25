from tensorflow.contrib import training as contrib_training
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers

class Network_model(object):
    def __init__(self, config):
        self.config = config
        self.inputs_tensor = tf.placeholder(tf.float32, [None, None, config.one_hot_length], name="inputs_tensor")
        self.labels_ = tf.placeholder(tf.float32, [None, 2], name="labels_")
        self.lengths = tf.placeholder(tf.int32, [None], name="lengths")
        self.lstmcell = tf.nn.rnn_cell.LSTMCell(num_units=config.num_units, state_is_tuple=True)
        self.network()

    def network(self):
        with tf.name_scope('LSTM'):
            self.initial_state = self.lstmcell.zero_state(self.config.batch_size, tf.float32)
            # initial_state = lstmcell.zero_state(batch_size, tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(
                self.lstmcell, self.inputs_tensor, sequence_length=self.lengths, initial_state=self.initial_state,
                swap_memory=True)
            self.final_out = final_state.h
        #
        with tf.name_scope('fc'):
            self.tem = contrib_layers.linear(self.final_out, 60)
            self.logi = contrib_layers.linear(self.tem, 30)
            # logits_flat = contrib_layers.linear(final_out, 20)
            self.logits_flat = contrib_layers.linear(self.logi, 2)
        with tf.name_scope('train'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_flat,
                                                                      labels=self.labels_))
            self.train_op = tf.train.AdamOptimizer(self.config.initial_learning_rate).minimize(self.loss)
        with tf.name_scope("result"):
            prediction = tf.nn.softmax(self.logits_flat)
            correct_predictions = tf.to_float(
            tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels_, 1)))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
