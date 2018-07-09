import json
import codecs
import numpy as np
import tensorflow as tf


class PrepareData(object):
    def __init__(self, image_path, ratio):
        self.imagePath = image_path
        self.ratio = ratio
        self.srcData = self.__load_china_tax()
        self.totalLen = len(self.srcData["imagePath"])

    def __load_china_tax(self):
        with codecs.open(self.imagePath, "r", "utf8") as f:
            return json.loads(f.read())

    @property
    def train_and_test_data(self):
        train_len = int(self.totalLen * self.ratio)
        rand_idx = np.random.permutation(self.totalLen)
        train_file_name = [self.srcData["imagePath"][idx] for idx in rand_idx[0:train_len]]
        test_file_name = [self.srcData["imagePath"][idx] for idx in rand_idx[train_len:]]
        train_label = [self.srcData["label"][idx] for idx in rand_idx[0:train_len]]
        test_label = [self.srcData["label"][idx] for idx in rand_idx[train_len:]]
        train_spare_label = tf.keras.preprocessing.sequence.pad_sequences(train_label, maxlen=6, dtype="int32",
                                                                          padding="post", truncating="post", value=-1)
        test_spare_label = tf.keras.preprocessing.sequence.pad_sequences(test_label, maxlen=6, dtype="int32",
                                                                         padding="post", truncating="post", value=-1)
        return train_file_name, train_spare_label.tolist(), test_file_name, test_spare_label.tolist()
