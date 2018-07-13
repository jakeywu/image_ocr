import os
import tensorflow as tf


class ImageCaptchaHandle(object):
    def __init__(self):
        self.CaptchaWidth = 192
        self.CaptchaHeight = 48

    def _image_process(self, image_path_and_target):
        image = self.__read_image(image_path_and_target["file_path"])
        image = self.format_image(image)
        label = image_path_and_target["target"]
        return image, label

    def format_image(self, image):
        shape = tf.shape(image)
        dst_width = tf.to_int32(shape[1] * self.CaptchaHeight / shape[0])
        image = tf.image.resize_images(image, tf.stack([self.CaptchaHeight, dst_width]))
        image = tf.image.resize_image_with_crop_or_pad(image, self.CaptchaHeight, self.CaptchaWidth)
        image = tf.image.transpose_image(image)
        image = 1. / 255 * image - 0.5
        return image

    @staticmethod
    def __read_image(image_path):
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.uint8)
        return image

    @staticmethod
    def _cnn_2d(neural, scope_name, in_channels, out_channels, filter_height=3, filter_width=3):
        """二维图像卷积"""
        with tf.variable_scope(name_or_scope=scope_name):
            kernel = tf.get_variable(
                name="W", shape=[filter_height, filter_width, in_channels, out_channels], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            bias = tf.get_variable(name="a", shape=[out_channels], dtype=tf.float32,
                                   initializer=tf.constant_initializer())
            con2d_op = tf.nn.conv2d(input=neural, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.nn.bias_add(con2d_op, bias=bias)

    @staticmethod
    def _batch_norm(neural, scope_name, is_training):
        """批量归一化"""
        with tf.variable_scope(scope_name):
            bn_neural = tf.layers.batch_normalization(
                inputs=neural, center=True, scale=True, training=is_training, name="batchNormalization")
            return bn_neural

    @staticmethod
    def _max_pool(neural, ksize):
        return tf.nn.max_pool(
            neural, ksize=(1, ksize, ksize, 1), strides=[1, 2, 2, 1], padding="SAME", name="max_pool"
        )

    @staticmethod
    def _get_sparse_tensor(inputs, space_value=-1):
        # 返回非-1的索引位置 (? * 2)
        idx = tf.where(tf.not_equal(inputs, tf.cast(space_value, inputs.dtype)))
        # 将参数中的切片收集到由索引指定的形状的张量中  ?
        val = tf.gather_nd(params=inputs, indices=idx)
        # (? * 8)
        shp = tf.shape(inputs, out_type=tf.int64)
        return tf.SparseTensor(indices=idx, values=val, dense_shape=shp)


class CnnRnnCtcOrc(ImageCaptchaHandle):
    def __init__(self, max_captcha_len, output_dim, num_lstm_hidden):
        super(CnnRnnCtcOrc, self).__init__()
        self.maxCaptchaLen = max_captcha_len
        self.outputDim = output_dim
        self.NumLSTMHidden = num_lstm_hidden
        self._init_placeholder()
        self._make_data_iterator()
        self.logits = self._inference(self.images, self.is_training)
        self._build_train_op()
        self.session = tf.Session()

    def _init_placeholder(self):
        self.file_paths = tf.placeholder(tf.string, shape=[None], name="image_names")
        self.targets = tf.placeholder(tf.int32, shape=[None, self.maxCaptchaLen])
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.batch_size = tf.placeholder(tf.int64, name="batch_size")
        self.epoch = tf.placeholder(tf.int64, name="epoch")

    def _make_data_iterator(self):
        dataset = tf.data.Dataset.from_tensor_slices({"file_path": self.file_paths, "target": self.targets})
        dataset = dataset.map(self._image_process).batch(self.batch_size).repeat(self.epoch)
        self.iterator = dataset.make_initializable_iterator()
        self.images, self.labels = self.iterator.get_next()

    def _inference(self, inputs, is_training):
        cnn_layers = self.__cnn_layers(inputs, is_training)  # 默认NWHC
        shapes = cnn_layers.get_shape().as_list()
        # 视W为特征max_timestep
        nets = tf.reshape(cnn_layers, shape=[-1, shapes[1], shapes[2] * shapes[3]])
        nets = tf.layers.dense(inputs=nets, units=256, activation=tf.nn.relu)
        self.seq_len = tf.reduce_sum(tf.cast(tf.not_equal(-9999999., tf.reduce_sum(nets, axis=2)),
                                             tf.int32), axis=1)
        return self.__lstm_layers(inputs=nets)

    def _build_train_op(self):
        self.sparse_label = self._get_sparse_tensor(self.labels)
        self.ctc_loss = tf.nn.ctc_loss(inputs=self.logits, labels=self.sparse_label, sequence_length=self.seq_len)
        self.loss = tf.reduce_mean(self.ctc_loss)
        self.ctc_decode_result = tf.nn.ctc_greedy_decoder(inputs=self.logits, sequence_length=self.seq_len)
        self.model_predict = tf.to_int32(self.ctc_decode_result[0][0], name="model_predict")
        edit_dist = tf.edit_distance(self.model_predict, self.sparse_label, normalize=True)
        self.distance = tf.reduce_mean(edit_dist)
        self.accuracy = tf.reduce_mean(tf.to_float(tf.less_equal(edit_dist, 0.)))

    def __lstm_layers(self, inputs):
        with tf.variable_scope("lstm"):
            shape = tf.shape(inputs)
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.NumLSTMHidden)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.NumLSTMHidden)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=self.seq_len, dtype=tf.float32)
            bi_outputs = tf.reshape(tf.concat(outputs, 2), [-1, 2 * self.NumLSTMHidden])
            lstm_w = tf.get_variable(
                name="W_out", shape=[2 * self.NumLSTMHidden, self.outputDim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            lstm_b = tf.get_variable(name="b_out", shape=[self.outputDim], dtype=tf.float32,
                                     initializer=tf.constant_initializer())

            self.logits = tf.matmul(bi_outputs, lstm_w) + lstm_b
            self.logits = tf.reshape(self.logits, [shape[0], inputs.shape[1], self.outputDim])
            # TimeStep major for ctc loss
            return tf.transpose(self.logits, (1, 0, 2))

    def __cnn_layers(self, inputs, is_training):
        """3个卷积层"""
        with tf.name_scope("cnn"):
            conv1 = self._cnn_2d(inputs, "cnn-1", 3, 32)
            bn1 = self._batch_norm(conv1, "bn-1", is_training)
            relu1 = tf.nn.leaky_relu(bn1, alpha=0.1, name="leaky_relu")
            pool1 = self._max_pool(relu1, 2)

            conv2 = self._cnn_2d(pool1, "cnn-2", 32, 64)
            bn2 = self._batch_norm(conv2, "bn-2", is_training)
            p_relu2 = tf.nn.leaky_relu(bn2, alpha=0.1, name="leaky_relu")
            pool2 = self._max_pool(p_relu2, 2)

            conv3 = self._cnn_2d(pool2, "cnn-3", 64, 128)
            bn3 = self._batch_norm(conv3, "bn-3", is_training)
            p_relu3 = tf.nn.leaky_relu(bn3, alpha=0.1, name="leaky_relu")
            pool3 = self._max_pool(p_relu3, 2)
        return pool3

    @staticmethod
    def save(session):
        saver = tf.train.Saver()
        checkpoints_path = "checkpoints/model"
        if os.path.exists(checkpoints_path):
            os.mkdir(checkpoints_path)
        saver.save(session, checkpoints_path)

    @staticmethod
    def load(session):
        checkpoints_path = "checkpoints/"
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(checkpoints_path))

    def train(self, train_x, train_y, config):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        self.session.run(tf.global_variables_initializer())
        self.session.run(
            [self.iterator.initializer],
            feed_dict={self.file_paths: train_x, self.targets: train_y, self.batch_size: config.batch_size, self.epoch: config.epoch}
        )
        step = 0
        while True:
            try:
                _, ls, dist, acc, s = self.session.run(
                    [train_op, self.loss, self.distance, self.accuracy, self.sparse_label],
                    feed_dict={self.is_training: True})
                print("**Train** STEP[%04d]: Loss: %.4f \t Distance: %.4f \t Accuracy: %.4f" % (step, ls, dist, acc))
                step += 1
            except tf.errors.OutOfRangeError:
                break
        self.save(self.session)

    def test(self, test_x, test_y, config):
        self.session.run(self.iterator.initializer,
                         feed_dict={self.file_paths: test_x,
                                    self.labels: test_y,
                                    self.epoch: 1,
                                    self.batch_size: config.batch_size})

        step = 0
        while True:
            try:
                ls, dist, acc = self.session.run([self.loss, self.distance, self.accuracy],
                                                 feed_dict={self.is_training: False})
                print("++Test+++ STEP[%04d]: Loss: %.4f \t Distance: %.4f \t Accuracy: %.4f" % (step, ls, dist, acc))
                step += 1
            except tf.errors.OutOfRangeError:
                break

    def release(self, export_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)
        g = tf.Graph()
        with g.as_default():
            input_images = tf.placeholder(tf.uint8, shape=[None, None, None, 3])
            f_input_images = tf.map_fn(ImageCaptchaHandle.format_image, input_images, tf.float32)
            outputs = self._inference(f_input_images, is_training=False)
            seq_length = tf.reduce_sum(tf.cast(tf.not_equal(-999999., tf.reduce_sum(outputs, axis=2)),
                                               tf.int32), axis=0)
            ctc_decode_result = tf.nn.ctc_greedy_decoder(inputs=outputs, sequence_length=seq_length)
            model_predict = tf.to_int32(ctc_decode_result[0][0])
            final_predict = tf.sparse_tensor_to_dense(model_predict, default_value=-1)
            with tf.Session(graph=g) as sess:
                self.load(sess)
                tensor_info_image = tf.saved_model.utils.build_tensor_info(input_images)
                tensor_info_predict = tf.saved_model.utils.build_tensor_info(final_predict)
                prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={"images": tensor_info_image},
                    outputs={"scores": tensor_info_predict},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
                builder.add_meta_graph_and_variables(
                    sess, tags=[tf.saved_model.tag_constants.SERVING],
                    signature_def_map={"predict_images": prediction_signature}
                )
                builder.save()
