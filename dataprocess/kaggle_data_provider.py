from .train_image_process import *
from .train_label_process import *
import numpy as np
from utility.tool import *
import os

split_factor = 0.6


class KaggleDataProvider(object):
    def __init__(self, image_size=64):
        # on Linux
        self.save_img_dir = "/home/steven/桌面/kaggle/data/clip_image.csv"
        self.save_mask_dir = "/home/steven/桌面/kaggle/data/clip_mask_image.csv"
        self.images = []
        self.labels = []
        self.train_images = []
        self.train_labels = []
        self.validate_images = []
        self.validate_labels = []
        data_exist_bool = os.path.exists(self.save_img_dir) & os.path.exists(self.save_mask_dir)

        if data_exist_bool:
            self.labels = load_dict_from_csv(self.save_mask_dir)
            self.images = load_dict_from_csv(self.save_img_dir)
            length_data = len(self.labels)
            train_length = int(split_factor * length_data)
            self.train_images = self.images[:length_data]
            self.train_labels = self.labels[:length_data]
            self.validate_images = self.images[length_data:]
            self.validate_labels = self.labels[length_data:]
            return

        self.img_size = image_size
        self.image_provider = TrainImageHandler()
        self.label_provider = TrainLabelsProcessor()

        self.dict_imageID_image = self.image_provider.dict_imgID_image
        self.dict_imageID_label = self.label_provider.dict_imgID_mask

        self.pad_w = 0
        self.pad_h = 0
        self.column = 1
        self.row = 1
        self.calculate_clip_size()
        self.get_train_val_data()

    def calculate_clip_size(self):
        image_width = 702
        image_height = 520

        threshold = round(0.5 * self.img_size)
        m_w = image_width % self.img_size
        m_h = image_height % self.img_size

        if m_w < threshold:
            self.column = image_width / self.img_size
        else:
            self.column = image_width / self.img_size + 1
            self.pad_w = self.img_size - m_w

        if m_h < threshold:
            self.row = image_height / self.img_size
        else:
            self.row = image_height / self.img_size + 1
            self.pad_h = self.img_size - m_h

    def get_train_val_data(self):
        for img_id, label in self.dict_imageID_label.items():
            # first padding
            label_array = np.array(label)
            label_pad = np.pad(label_array, ((0, self.pad_h), (0, self.pad_w)), 'constant', constant_values=(0, 0))
            label = label_pad.tolist()
            image = self.dict_imageID_image[img_id]
            image_array = np.array(image)
            image_pad = np.pad(image_array, ((0, self.pad_h), (0, self.pad_w)), 'constant', constant_values=(0, 0))
            image = image_pad.tolist()

            # clip image and mask
            for r in range(self.row):  # first dimension
                start_r = r * self.img_size
                end_r = start_r + self.img_size
                for col in range(self.column):  # second dimension
                    start_col = col * self.img_size
                    end_col = col * self.img_size
                    mask_clip = label[start_r:end_r][start_col:end_col]
                    image_clip = image[start_r:end_r][start_col:end_col]
                    self.labels.append(mask_clip)
                    self.images.append(image_clip)

        save_dict_as_csv(self.save_mask_dir, self.labels)
        save_dict_as_csv(self.save_img_dir, self.images)
        length_data = len(self.labels)
        train_length = int(split_factor * length_data)
        self.train_images = self.images[:length_data]
        self.train_labels = self.labels[:length_data]
        self.validate_images = self.images[length_data:]
        self.validate_labels = self.labels[length_data:]
