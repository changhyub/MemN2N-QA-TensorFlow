import os
import pprint
import tensorflow as tf

from data import read_data
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 100, "internal state dimension [100]")
flags.DEFINE_integer("nhop", 2, "number of hops [2]")
flags.DEFINE_integer("mem_size", 20, "memory size [20]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 20, "number of epoch to use during training [20]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string("data_name", "qa1", "data set name qa#")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
# flags.DEFINE_string("log_dir", "./board/test3", "log dir for tenserboard")

FLAGS = flags.FLAGS

def main(_):
    count = []
    word2idx = {}

    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)

    train_data, train_query, train_target, train_idx = read_data('%s/%s_train.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx, FLAGS.mem_size)
    valid_data, valid_query, valid_target, valid_idx = read_data('%s/%s_valid.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx, FLAGS.mem_size)
    test_data, test_query, test_target, test_idx = read_data('%s/%s_test.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx, FLAGS.mem_size)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))

    max = 0
    for i in train_data, valid_data, test_data:
        if len(i) > max:
            max = len(i)

    FLAGS.max_len = max # the longest sentence length
    FLAGS.nwords = len(word2idx)

    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()

        if FLAGS.is_test:
            model.run(valid_data, valid_query, valid_target, valid_idx, test_data, test_query, test_target, test_idx)
        else:
            model.run(train_data, train_query, train_target, train_idx, valid_data, valid_query, valid_target, valid_idx)

if __name__ == '__main__':
    tf.app.run()
