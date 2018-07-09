import os
import json
import codecs
import tensorflow as tf


class PreImageByTensorSlice(object):
    def __init__(self, file_path):
        self.srcPath = file_path
        self.vocabIndex = self.__read_chinese_vocab()

    def read_captcha_files(self):
        if not tf.gfile.IsDirectory(self.srcPath):
            assert "{}目录不存在".format(self.srcPath)
        file_names = tf.gfile.ListDirectory(self.srcPath)
        if not file_names:
            assert "{}目录下文件不能为空".format(self.srcPath)
        return file_names

    @staticmethod
    def __read_chinese_vocab():
        vocab_index = dict()
        chinese_vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chinese_vocab.txt")
        with codecs.open(chinese_vocab_path, "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                vocab_index[line.replace("\n", "")] = i
        return vocab_index

    def format_china_tax(self):
        """国家税务网"""
        image_dict = dict()
        label_lst = []
        image_names = self.read_captcha_files()
        for file_name in image_names:
            try:
                captcha_char = file_name.split(".")[0].split("_")[2]
                print(captcha_char)
                char_index = [self.vocabIndex[char.lower()] for char in captcha_char]
                label_lst.append(char_index)
            except Exception as e:
                raise e
        image_dict["label"] = label_lst
        image_dict["imagePath"] = [os.path.join(self.srcPath, image_name) for image_name in image_names]
        return image_dict


if __name__ == "__main__":
    pbts = PreImageByTensorSlice("/home/littlefish/trainData/image/captcha/china_tax")
    with codecs.open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "china_tax.txt"), "a", "utf8") as f:
        f.write(json.dumps(pbts.format_china_tax(), indent=2, ensure_ascii=False))
