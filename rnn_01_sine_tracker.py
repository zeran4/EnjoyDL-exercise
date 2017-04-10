"""
* this is rnn example
"""

import tensorflow as tf

CONSTANT = tf.app.flags
CONSTANT.DEFINE_integer("epoch", 1000, "epoch when learning")
CONSTANT.DEFINE_integer("samples", 1000, "simulation data samples")
CONSTANT.DEFINE_integer("hidden", 5, "hidden layers in rnn")
CONSTANT.DEFINE_integer("vec_size", 1, "input vector size into rnn")
CONSTANT.DEFINE_integer("batch_size", 10, "minibatch size for training")
CONSTANT.DEFINE_integer("state_size", 15, "state size in rnn")
CONSTANT.DEFINE_integer("recurrent", 5, "recurrent step")
CONSTANT.DEFINE_float("learning_rate", 0.01, 'learning rate for optimizer')
CONSTANT.DEFINE_string("ckpt_dir", "./checkpoint/rnn", "check point log dir")
CONSTANT.DEFINE_string("tensorboard_dir", "./tensorboard", "tensorboard log dir")
CONST = CONSTANT.FLAGS

class RNN(object):
    """
     * RNN model
    """
    def __init__(self):
        self._to_plot()
        print("ready to visualize")
        self._gen_sim_data()
        print("generated data")
        self._build_batch()
        print("batch built")
        #self._set_variables()
        print("variables set")
        self._build_model()
        print("model built")
        self._save_model()
        print("saver created")
        self._build_train()
        print("loss graph created")
        self._initialize()
        print("initialized")

    def training(self):
        """
        * run prediction
        """
        print("start training....")
        
        self.sess.run(tf.global_variables_initializer())

        for step in range(CONST.epoch):
            _, loss = self.sess.run([self.train, self.loss])
            if step % 10 == 0:
                print("step: ", step)
                print("loss: ", loss)

            if step % 100 == 0:
                self._write_checkpoint(CONST.ckpt_dir)
                print("model saved...")

        print("training done")

    def prediction(self):
        """
         * run training
        """
        self._run_pred()
        self._line_plot_draw(100)
        self._close_session()

    @classmethod
    def _run_train(cls):
        cls.sess.run(tf.global_variables_initializer())
        _, loss = cls.sess.run([cls.train, cls.loss])
        return loss

    @classmethod
    def _run_pred(cls):
        cls._restore_checkpoint(CONST.ckpt_dir)
        return cls.sess.run(cls.pred)

    @classmethod
    def _save_model(cls):
        cls.saver = tf.train.Saver()

    @classmethod
    def _write_checkpoint(cls, directory):
        cls.saver.save(cls.sess, directory)

    @classmethod
    def _restore_checkpoint(cls, directory):
        cls.saver.restore(cls.sess, directory)

    @classmethod
    def _initialize(cls):
        cls.sess = tf.Session()
        cls.coord = tf.train.Coordinator()
        cls.thread = tf.train.start_queue_runners(cls.sess, cls.coord)

    @classmethod
    def _close_session(cls):
        cls.coord.request_stop()
        cls.coord.join(cls.thread)
        cls.sess.close()

    @classmethod
    def _gen_sim_data(cls):
        cls.ts_x = tf.constant([i for i in range(CONST.samples+1)], dtype=tf.float32)
        ts_y = tf.sin(cls.ts_x * 0.1)

        sp_batch = (int(CONST.samples/CONST.hidden), CONST.hidden, CONST.vec_size)
        cls.batch_input = tf.reshape(ts_y[:-1], sp_batch)
        cls.batch_label = tf.reshape(ts_y[1:], sp_batch)

        cls._line_plot("sin_x", ts_y)

    @classmethod
    def _build_batch(cls):
        batch_set = [cls.batch_input, cls.batch_label]
        cls.b_train, cls.b_label = tf.train.batch(batch_set, CONST.batch_size, enqueue_many=True)

    @classmethod
    def _set_variables(cls):
        linear_w = tf.Variable(tf.truncated_normal([CONST.recurrent, CONST.state_size, CONST.input_vector_size]))
        linear_b = tf.Variable(tf.zeros([CONST.recurrent, 1, 1]))

        cls.linear_w = tf.unstack(linear_w)
        cls.linear_b = tf.unstack(linear_b)

    @classmethod
    def _build_model(cls):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(CONST.state_size)
        cls.input_set = tf.unstack(cls.b_train, axis=1)
        cls.label_set = tf.unstack(cls.b_label, axis=1)

        cls.output, _ = tf.contrib.rnn.static_rnn(rnn_cell, cls.input_set, dtype=tf.float32)

        cls.output_w = tf.Variable(tf.truncated_normal([CONST.hidden, CONST.state_size, CONST.vec_size]))
        output_b = tf.Variable(tf.zeros([CONST.vec_size]))

        cls.pred = tf.matmul(cls.output, cls.output_w ) + output_b
        # print("output_w: ", cls.sess.run(tf.shape(cls.output_w)))
        # print("output: ", cls.sess.run(tf.shape(cls.output)))

        #cls._line_plot("1_label_set", tf.transpose(cls.label_set, (1, 0, 2)))
        #cls._line_plot("2_pred_sin", tf.transpose(cls.pred, (1, 0, 2)))

    @classmethod
    def _build_train(cls):
        cls.loss = 0
        for i in range(CONST.hidden):
            cls.loss += tf.losses.mean_squared_error(tf.unstack(cls.b_label, axis=1)[i], cls.pred[i])

        cls.train = tf.train.AdamOptimizer(CONST.learning_rate).minimize(cls.loss)

    # 직접 만든 loss function을 사용해도 됨 : 여기서는 tensorflow에 있는 것을 사용    
    @classmethod
    def _mean_square_error(cls, batch, label):
        return tf.reduce_mean(tf.pow(batch - label, 2))

    # @classmethod
    # def _line_plot(cls, name, tensor):
    #     ts_vector = tf.reshape(tf.stack(tensor), (CONST.batch_size*CONST.recurrent,))
    #     plot = ts_vector[cls.idx]
    #     tf.summary.scalar(name, plot)

    @classmethod
    def _line_plot(cls, name, tensor):
        plot = tensor[cls.idx]
        tf.summary.scalar(name, plot)

    @classmethod
    def _print(cls, tensor):
        with tf.Session() as sess:
            print(sess.run(tensor))

    @classmethod
    def _line_plot_draw(cls, num_plot):
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(CONST.tensorboard_dir)
        for i in range(num_plot):
            summary_str = cls.sess.run(summaries, {cls.idx: i})
            writer.add_summary(summary_str, i)
            writer.flush()
        writer.close()

    @classmethod
    def _to_plot(cls):
        cls.idx = tf.placeholder(tf.int32)

def main(_):
    """
    * code start here
    """
    print("code start")
    rnn = RNN()
    rnn.training()
    rnn.prediction()
    print("end process")

if __name__ == "__main__":
    tf.app.run()
    