from models.inception_v3 import InceptionV3
import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.utils import to_categorical


def test_model():
    input = np.random.randn(10, 224, 224, 3)
    logging.debug(input.shape)
    label = to_categorical(np.random.randint(1000, size=10), 1000)

    model = InceptionV3()
    model.train_model(input, label, 1000, 32)


def main(unused_argvs):
    test_model()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
