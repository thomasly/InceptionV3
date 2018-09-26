import tensorflow as tf
import logging


def conv2d(x, kernel_size, filters, strides=1, name='conv2d'):
    """ Easy conv2d with stride=1 and padding='same'
    """
    logging.debug("x shape of {}: {}".format(name, x.get_shape().as_list()))
    with tf.name_scope(name):
        return tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same'
        )


def inception_module(
    x, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, n_pool, name='inception'
):
    """ Incetion module of GoogLeNet
    Arguments:
        x - input, 4D tensor
        n1x1 - integer, number of 1x1 filters
        n3x3_reduce - integer, number of 1x1 filters before 3x3 conv
        n3x3 - integer, number of 3x3 filters
        n5x5_reduce - integer, number of 1x1 filters before 5x5 conv
        n5x5 - integer, number of 5x5 filters
        name - string

    Return:
        4D tensor
    """
    with tf.name_scope(name=name):
        output_1x1 = conv2d(x, 1, n1x1)
        output_1x1 = tf.nn.relu(output_1x1)

        reduce_3x3 = conv2d(x, 1, n3x3_reduce)
        output_3x3 = conv2d(reduce_3x3, 3, n3x3)
        output_3x3 = tf.nn.relu(output_3x3)

        reduce_5x5 = conv2d(x, 1, n5x5_reduce)
        output_5x5 = conv2d(reduce_5x5, 5, n5x5)
        output_5x5 = tf.nn.relu(output_5x5)

        out_pooling = tf.layers.max_pooling2d(x, 3, 1, 'SAME')
        out_pooling = conv2d(out_pooling, 1, n_pool, name='pooling')

        inception_output = tf.concat(
            [output_1x1, output_3x3, output_5x5, out_pooling], 3
        )
        return inception_output
