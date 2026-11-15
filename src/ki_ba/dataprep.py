import numpy as np
import math

class DataPrep:
    def __init__(
            self, raw_images, raw_labels, class_count, 
            scale_factor, validation_size, seed,
            raw_test_images = None, raw_test_labels = None
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

        if raw_test_images is not None and raw_test_labels is not None:

            if len(raw_test_images) != len(raw_test_labels):
                raise ValueError("Raw Test Labels and Raw Test Images need to be the same length")
            
            self.test_images = self._shape_and_normalize_raw_images(raw_test_images)
            self.test_labels = self._one_hot(raw_test_labels)
        else:
            self.test_images = None
            self.test_labels = None

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
        idx = self.rng.permutation(len(images))

        val_idx = idx[:self.validation_size]
        train_idx = idx[self.validation_size:]

        train_images = images[train_idx]
        train_labels = one_hot[train_idx]
        validation_images = images[val_idx]
        validation_labels = one_hot[val_idx]

        return train_images, train_labels, validation_images, validation_labels

    def get_mini_batches(self, batch_size):
        amount_of_batches = math.ceil(len(self.train_images) / batch_size)
        batches_training = []
        batch_size_sum = 0

        idx = self.rng.permutation(len(self.train_images))
        shuffled_images = self.train_images[idx]
        shuffled_labels = self.train_labels[idx]

        for i in range(0, amount_of_batches):
            mini_batch = (
                shuffled_images[batch_size_sum:batch_size_sum + batch_size],
                shuffled_labels[batch_size_sum:batch_size_sum + batch_size]
            )

            batches_training.append(mini_batch)

            batch_size_sum = batch_size_sum + batch_size

        return batches_training

    def get_validation_data(self):
        return self.validation_images, self.validation_labels

    def get_test_data(self):
        return self.test_images, self.test_labels