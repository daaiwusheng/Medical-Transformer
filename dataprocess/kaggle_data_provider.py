from .train_image_process import *
from .train_label_process import *

split_factor = 0.6

class KaggleDataProvider(object):
    def __init__(self):
        self.image_provider = TrainImageHandler()
        self.label_provider = TrainLabelsProcessor()

        self.dict_imageID_image = self.image_provider.dict_imgID_image
        self.dict_imageID_label = self.label_provider.dict_imgID_mask

        self.images = []
        self.labels = []

        self.train_images = []
        self.train_labels = []

        self.validate_images = []
        self.validate_labels = []

    def get_train_val_data(self):
        for img_id, label in self.dict_imageID_label.items():
            self.labels.append(label)
            self.images.append(self.dict_imageID_image[img_id])

        length_data = len(self.labels)
        train_length = int(split_factor * length_data)
        self.train_images = self.images[:length_data]
        self.train_labels = self.labels[:length_data]
        self.validate_images = self.images[length_data:]
        self.validate_labels = self.labels[length_data:]
