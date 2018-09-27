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
    model.train_model(input, label, 100, 32)


def cifar10_test():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[0:200, ...]
    y_train = to_categorical(y_train)[0:200, ...]
    logging.debug('y_train shape: {}'.format(y_train.shape))
    y_test = to_categorical(y_test)
    model = InceptionV3(input_shape=(None, 32, 32, 3), classes=10)
    model.train_model(x_train, y_train, 100, 32)


def main(unused_argvs):
    cifar10_test()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    tf.app.run()
