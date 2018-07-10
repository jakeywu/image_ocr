import tensorflow as tf
from captcha.utils import PrepareData
from captcha.cnn_lstm_ctc_ocr_model import CnnRnnCtcOrc

tf.flags.DEFINE_string(name="image_path", default="/usr/projects/nlp/imageRecognition/data/china_tax.txt", help="image path which had prepared")
tf.flags.DEFINE_string(name="mode", default="train", help="train, test")
tf.flags.DEFINE_float(name="ratio", default=0.9, help="train ratio of all data")
tf.flags.DEFINE_integer(name="max_captcha_len", default=6, help="max captcha len")
tf.flags.DEFINE_integer(name="output_dim", default=11, help="the total length is 11:  0-9 and blank ")
tf.flags.DEFINE_integer(name="num_lstm_hidden", default=64, help="multi lstm num hidden")
tf.flags.DEFINE_integer(name="batch_size", default=64, help="batch size of train data ")
tf.flags.DEFINE_integer(name="epoch", default=100, help="maximum epochs")
tf.flags.DEFINE_float(name="learning_rate", default=0.01, help="inital learning rate")

FLAGS = tf.flags.FLAGS


def main(_):
    train_x, train_y, test_x, test_y = PrepareData(FLAGS.image_path, FLAGS.ratio).train_and_test_data
    model = CnnRnnCtcOrc(FLAGS.max_captcha_len, FLAGS.output_dim, FLAGS.num_lstm_hidden)
    model.train(train_x, train_y, FLAGS)


if __name__ == "__main__":
    tf.app.run(main)
