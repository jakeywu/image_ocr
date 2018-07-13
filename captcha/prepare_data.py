import os
import cv2
import json
import codecs
import collections
from collections import Counter

color_range = {
    "Red": ((200, 255), (0, 150), (0, 150)),
    "Blue": ((0, 150), (0, 150), (200, 255)),
    "Yellow": ((200, 255), (200, 255), (0, 150))
}


class PreImageByTensorSlice(object):
    def __init__(self, src, target):
        self.srcPath = src
        self.targetPath = target
        self.__del_target_file_names()
        self.fileNames = self.__image_file_names()
        self.charPath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "char_index.txt")
        self.charIndex = self.__captcha_char()

    def __image_file_names(self):
        return os.listdir(self.srcPath)

    def __del_target_file_names(self):
        for name in os.listdir(self.targetPath):
            os.remove(os.path.join(self.targetPath, name))

    def __captcha_char(self):
        char_index = collections.OrderedDict()
        index_char = collections.OrderedDict()
        char_names = "".join([str(file_name.split(".")[0]).split("_")[2].lower() for file_name in self.fileNames])
        for index, char in enumerate(list(set(char_names))):
            char_index[char.lower()] = index
            index_char[index] = char.lower()
        with codecs.open(self.charPath, "w", "utf8") as f:
            f.write(json.dumps(index_char, indent=2, ensure_ascii=False))
        return char_index

    def format_china_tax(self):
        """国家税务网"""
        image_dict = dict()
        label_lst = []
        for file_name in self.fileNames:
            try:
                captcha_char = str(file_name.split(".")[0]).split("_")[2].lower()
                char_index = [self.charIndex[char] for char in captcha_char]
                label_lst.append(char_index)
            except Exception as e:
                raise e

        image_dict["label"] = label_lst
        image_dict["imagePath"] = [os.path.join(self.targetPath, image_name) for image_name in self.fileNames]
        return image_dict

    @staticmethod
    def __check_pixel(pixel, s_color):
        """
        pixel:  list
        """
        if not s_color[0][0] <= pixel[0] <= s_color[0][1]:
            return False
        if not s_color[1][0] <= pixel[1] <= s_color[1][1]:
            return False
        if not s_color[2][0] <= pixel[2] <= s_color[2][1]:
            return False
        return True

    def convert_image_color(self, img, color):
        target_pixels = []
        given_pixels = []
        assert color in ["Red", "Blue", "Yellow", "All"]
        if color == "All":
            return img
        s_color = color_range[color]
        for pixel in img.reshape(-1, 3):
            pixel = pixel.tolist()
            if self.__check_pixel(pixel, s_color):
                given_pixels.append(pixel)
            target_pixels.append(" ".join([str(p) for p in pixel]))

        target = [int(p) for p in Counter(target_pixels).most_common(1)[0][0].split(" ")]

        for col in range(img.shape[0]):
            for row in range(img.shape[1]):
                px = img[col, row, :].tolist()
                if px in given_pixels:
                    continue
                img[col, row, :] = target
        return img

    def __img_convert(self, file_path):
        try:
            src_name = os.path.basename(file_path)
            color = src_name.split("_")[1].capitalize()
            img = cv2.imread(filename=file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.convert_image_color(img, color)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.targetPath, src_name), img)
        except Exception as e:
            print(e)

    def prepare_img_data(self):
        if not os.path.exists(self.srcPath) or not os.path.exists(self.targetPath):
            raise Exception("srcPath or targetPath not exist")
        for file_path in os.listdir(self.srcPath):
            self.__img_convert(os.path.join(self.srcPath, file_path))
        china_tax = self.format_china_tax()
        with codecs.open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "china_tax.txt"), "w",
                         "utf8") as f:
            f.write(json.dumps(china_tax, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    src_path = "/home/littlefish/trainData/image/china_tax/source_path"
    target_path = "/home/littlefish/trainData/image/china_tax/target_path"
    PreImageByTensorSlice(src_path, target_path).prepare_img_data()
