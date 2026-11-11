import numpy as np

class DataPrep:
    def __init__(
            self, raw_images, raw_labels, class_count, 
            scale_factor, validation_size, seed
            ):
        
        self.class_count = class_count
        self.scale_factor = scale_factor
        self.validation_size = validation_size

        if len(raw_labels) != len(raw_images):
            raise ValueError("Raw Labels and Raw Images need to be the same length")

        self.rng = np.random.default_rng(seed)

        one_hot = self._one_hot(raw_labels)

        images = self._shape_and_normalize_raw_images(raw_images)

        self.train_images, self.train_labels, self.validation_images, self.validation_labels = self._split(images, one_hot)

    def _one_hot(self, raw_labels):
        one_hot = np.zeros((len(raw_labels), self.class_count))

        for i in range(0, len(raw_labels)):
            number = raw_labels[i]
            one_hot[i, number] = 1

        return one_hot
    

    def _shape_and_normalize_raw_images(self, raw_images):
        reshaped_images = np.reshape(raw_images, (len(raw_images), -1))

        normalized_images = reshaped_images / self.scale_factor

        return normalized_images

    def _split(self, images, one_hot):
        pass

    def get_mini_batches(self, batch_size):
        pass

    def get_validation_data(self):
        pass