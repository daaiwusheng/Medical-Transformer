from utility.rle_tool import *
from utility.tool import *
import os


class TrainImageHandler(object):
    def __init__(self):
        #on Mac
        self.img_dir = "/Users/wangyu/Desktop/利兹上课资料/Kaggle比赛/data/train";
        #on Linux
        self.img_dir = "/home/steven/桌面/kaggle/data/train"
        self.dict_imgID_image = {}
        self.get_dict_imgID_image()

    def __len__(self):
        return len(self.dict_imgID_mask)

    def get_dict_imgID_image(self):
        for root, dirs, files in os.walk(self.img_dir, topdown=True):
            for image_name in files:
                if '.png' in image_name:
                    image_full_path = os.path.join(self.img_dir, image_name)
                    img = cv2.imread(image_full_path, 0)
                    image_id = image_name.split('.')[0]
                    self.dict_imgID_image[image_id] = img



