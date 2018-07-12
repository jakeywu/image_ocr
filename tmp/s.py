import sys
import os
import uuid

path = "/home/littlefish/trainData/image/captcha/validate_num"

captcha_char = "0123456789"

# 523250049.18_8732.jpg
labels = []
path1 = []
for item in os.listdir(path):
    path1.append(os.path.join(path, item))
    labels.append([captcha_char.index(char) for char in item.split("_")[1].split(".")[0]])

import codecs
import json
with codecs.open("/usr/projects/nlp/imageRecognition/data/china_tax.txt", "w", "utf8") as f:
    f.write(json.dumps({"imagePath": path1, "label": labels}, indent=2, ensure_ascii=False))