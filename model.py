import os
import math
import random
import numpy as np
import tensorflow as tf
from past.builtins import xrange

tf.set_random_seed(777)

class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.max_grad_norm = config.max_grad_norm
        self.max_len = config.max_len
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.input = tf.placeholder(tf.int32, [self.batch_size, self.max_len], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size, self.max_len], name="context")

        self.hid = []
        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.rightCnt = None
        self.accuracy = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []

        # self.writer = tf.summary.FileWriter(config.log_dir, sess.graph)

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))

        # Temporal Encoding
        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
        self.T_B = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        Ain_c = tf.reduce_sum(Ain_c, 2)
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        Bin_c = tf.reduce_sum(Bin_c, 2)
        Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        Bin = tf.add(Bin_c, Bin_t)

        # Query 처리
        Uin = tf.nn.embedding_lookup(self.A, self.input)
        Uin = tf.reduce_sum(Uin, 1)
        self.hid.append(Uin)

        for h in xrange(self.nhop):
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.matmul(self.hid3dim, Ain, adjoint_b=True)
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
            P = tf.nn.softmax(Aout2dim)

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Bout = tf.matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.edim])

            Cout = tf.matmul(self.hid[-1], self.C)
            Dout = tf.add(Cout, Bout2dim)

            self.hid.append(Dout)

    def build_model(self):
        self.build_memory()

        self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], stddev=self.init_std))
        z = tf.matmul(self.hid[-1], self.W)

        self.rightCnt = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(z, 1), tf.argmax(self.target, 1)), tf.float32))

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.target)

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]

        grads_and_vars = self.opt.compute_gradients(self.loss,params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def train(self, data, query, answer, index):
        N = int(math.ceil(len(query) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.max_len])
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size, self.max_len])

        x.fill(0) #query
        context.fill(0)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        for idx in xrange(N):
            target.fill(0)
            for b in xrange(self.batch_size):
                m = random.randrange(len(query))
                toIdx = len(query[m])
                x[b, 0:toIdx] = query[m]
                for ans in answer[m]:
                    target[b][ans] = 1

                toIdx = index[m]
                for i in range(self.mem_size):
                    temp = data[toIdx - self.mem_size + i]
                    context[b,i,0:len(temp)] = temp[:]

            _, loss, self.step = self.sess.run([self.optim,
                                                self.loss,
                                                self.global_step],
                                                feed_dict={
                                                    self.input: x,
                                                    self.time: time,
                                                    self.target: target,
                                                    self.context: context})
            cost += np.sum(loss)

        return cost/N/self.batch_size

    def test(self, data, query, answer, index, label='Test'):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.max_len])
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size, self.max_len])

        x.fill(0) #query
        context.fill(0)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        m = 0
        rightCnt = 0
        for idx in xrange(N):
            target.fill(0)
            for b in xrange(self.batch_size):
                toIdx = len(query[m])
                x[b, 0:toIdx] = query[m]
                for ans in answer[m]:
                    target[b][ans] = 1

                toIdx = index[m]

                for i in range(self.mem_size):
                    temp = data[toIdx - self.mem_size + i]
                    context[b,i,0:len(temp)] = temp[:]

                m += 1
                if m >= len(query):
                    m = 0

            loss, cnt = self.sess.run([self.loss, self.rightCnt], feed_dict={self.input: x,
                                                         self.time: time,
                                                         self.target: target,
                                                         self.context: context})
            cost += np.sum(loss)
            rightCnt += cnt;

        self.accuracy = rightCnt / N / self.batch_size
        return cost/N/self.batch_size

    def run(self, train_data, train_query, train_target, train_idx, test_data, test_query, test_target, test_idx):
        if not self.is_test:
            for idx in xrange(self.nepoch):
                train_loss = np.sum(self.train(train_data, train_query, train_target, train_idx))
                test_loss = np.sum(self.test(test_data, test_query, test_target, test_idx, label='Validation'))

                # Logging
                self.log_loss.append([train_loss, test_loss])

                state = {
                    'train_error': math.exp(train_loss),
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'accuracy': self.accuracy
                }
                print(state)

                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.15
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step = self.step.astype(int))
        else:
            self.load()
            valid_loss = np.sum(self.test(train_data, label='Validation'))
            test_loss = np.sum(self.test(test_data, label='Test'))
            test_accuracy = self.accuracy

            state = {
                'valid_error': math.exp(valid_loss),
                'test_error': math.exp(test_loss),
                'test_accuracy': test_accuracy
            }
            print(state)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Test mode but no checkpoint found")
