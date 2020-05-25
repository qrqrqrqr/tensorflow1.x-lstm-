from tensorflow.contrib import training as contrib_training
import tensorflow.compat.v1 as tf
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


config = lstm_config()
# 数据部分
path = r''
temp = np.load(path)
train_inputs = temp["train_inputs"]
train_labels = temp["train_labels"]
input_len = temp["input_len"]

train_labels = np.array(train_labels)
input_len = np.array(input_len)
train_inputs = np.array(train_inputs)
dataset = tf.data.Dataset.from_tensor_slices((np.array(train_inputs), np.array(train_labels), np.array(input_len)))
batch_size = tf.placeholder(tf.int64, shape=[])
dataset = dataset.batch(config.batch_size)
dataset = dataset.shuffle(1000).repeat()
iterator = dataset.make_one_shot_iterator()
_inputs, _labels, _i_len = iterator.get_next()

# 网络部分
model_train = models.Network_model(config)

# train
saver = tf.train.Saver()
init = tf.initialize_all_variables()
with tf.Session() as sess:
    # config.batch_size
    tem_step_plt = []
    tem_loss_plt = []
    tem_acc_plt = []
    sess.run(init)
    for j in range(30000):
        i_, l, le = sess.run([_inputs, _labels, _i_len])
        lo_, _, acc_ = sess.run([model_train.loss, model_train.train_op, model_train.accuracy], {model_train.inputs_tensor: i_, model_train.labels_: l,
                                                             model_train.lengths: le})
        if j % 200 == 0:
            print(lo_, acc_)
            tem_loss_plt.append(lo_)
            tem_step_plt.append(j)
            tem_acc_plt.append(acc_)
            if j % 2000 == 0:
                saver.save(sess, r"model")
    np.savez(r'result.npz',tem_loss_plt=tem_loss_plt,tem_step_plt=tem_step_plt,tem_acc_plt=tem_acc_plt)



