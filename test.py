from tensorflow.contrib import training as contrib_training
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
import numpy as np

import model_note as models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def lstm_config():
    return contrib_training.HParams(
        use_dynamic_rnn=True,
        num_units = 100,
        batch_size=256,
        lr=0.0002,
        l2_reg=2.5e-5,
        clip_norm=5,
        initial_learning_rate=0.005,
        decay_steps=1000,
        decay_rate=0.85,
        rnn_layer_sizes=[100],
        skip_first_n_losses=32,
        one_hot_length=38,
        exponentially_decay_learning_rate=True)

temp = np.load(r'')
train_inputs = temp["train_inputs"]
train_labels = temp["train_labels"]
input_len = temp["input_len"]
dataset = tf.data.Dataset.from_tensor_slices((np.array(train_inputs), np.array(train_labels), np.array(input_len)))
dataset = dataset.batch(256)
iterator = dataset.make_one_shot_iterator()
_inputs, _labels, _i_len = iterator.get_next()

config = lstm_config()
model_train = models.Network_model(config)
with tf.Session() as sess:
    
    saver = tf.train.Saver()
    gragh = tf.get_default_graph()
    sum = 0
    count = 0
    saver.restore(sess, r'')
    while True:
         try:
            i_, l, le = sess.run([_inputs, _labels, _i_len])
            loss, acc, lo = sess.run([model_train.loss,  model_train.accuracy, model_train.logits_flat], {model_train.inputs_tensor: i_, model_train.labels_: l,
                                                                 model_train.lengths: le})
            _t5, _final_state = tf.nn.dynamic_rnn(
                model_train.lstmcell, i_, sequence_length=le, initial_state=model_train.initial_state,
                swap_memory=True)
            _t1 = contrib_layers.linear(_final_state, 60)
            _t2 = contrib_layers.linear(_t1, 30)
            _t3 = contrib_layers.linear(_t2, 2)
            _loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_t3,
                                                                      labels=l))
            prediction = tf.nn.softmax(_t3)
            correct_predictions = tf.to_float(
            tf.equal(tf.argmax(prediction, 1), tf.argmax(l, 1)))
            _accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            sum = sum + acc*config.batch_size
            count = count + config.batch_size
            print("acc is "+str(acc)+"  loss is  "+str(loss)+"   _loss is "+str(_loss)+"  _acc is "+str(_accuracy))
         except:
             print("accuracy is " + str(sum/count))
             break
