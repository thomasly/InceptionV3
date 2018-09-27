import os
import logging
import tensorflow as tf
from layers.inception_module import inception_module, conv2d
from datetime import datetime


class InceptionV3():
    def __init__(
        self, input_shape=(None, 224, 224, 3), classes=1000, is_train=True
    ):
        self.input = tf.placeholder(
            tf.float32, shape=input_shape, name='input'
        )
        self.is_train = is_train
        self.classes = classes
        self.model = self.build_model(
            self.input, is_train=is_train, classes=classes
        )
        self.timestamp = datetime.now().strftime(r'%Y%m%d_%H%M%S')

    def build_model(self, input, is_train=True, classes=1000):
        def model(input=input, is_train=is_train, classes=classes):
            with tf.name_scope('pre_inception'):
                # first convolution
                logging.debug(input.get_shape().as_list())
                logging.debug(input.get_shape().ndims)
                conv_1 = conv2d(input, 7, 64, 1, 'conv_1')
                output = tf.layers.max_pooling2d(
                    conv_1, 3, 1, 'same', name='max_pooling_1'
                )

                # second convolution
                reduce_1x1 = conv2d(output, 3, 64, name='reduce_1')
                output = conv2d(reduce_1x1, 3, 192, name='conv_2')
                output = tf.layers.max_pooling2d(
                    output, 3, 1, 'same', name='max_pooling_2'
                )

            with tf.name_scope('inception_block_1'):
                # inception modules block 1
                output = inception_module(
                    output, 64, 96, 128, 16, 32, 32, name='inception_3a'
                )
                output = inception_module(
                    output, 128, 128, 192, 32, 96, 64, name='inception_3b'
                )

            output = tf.layers.max_pooling2d(
                output, 3, 2, 'same', name='max_pooling_3'
            )

            with tf.name_scope('inception_block_2'):
                # inception modules block 2
                output = inception_module(
                    output, 192, 96, 208, 16, 48, 64, name='inception_4a'
                )
                output = inception_module(
                    output, 160, 112, 224, 24, 64, 64, name='inception_4b'
                )
                # output = inception_module(
                #     output, 128, 128, 256, 24, 64, 64, name='inception_4c'
                # )
                # output = inception_module(
                #     output, 112, 144, 288, 32, 64, 64, name='inception_4d'
                # )
                output = inception_module(
                    output, 256, 160, 320, 32, 128, 128, name='inception_4e'
                )
            output = tf.layers.max_pooling2d(
                output, 3, 2, 'same', name='max_pooling_4'
            )

            with tf.name_scope('inception_block_3'):
                # inception modules block 3
                output = inception_module(
                    output, 256, 160, 320, 32, 128, 128, name='inception_5a'
                )
                # output = inception_module(
                #     output, 384, 192, 384, 48, 128, 128, name='inception_5b'
                # )

            output = tf.layers.average_pooling2d(
                output, 7, 1, 'same', name='avg_pooling'
            )

            with tf.name_scope('dense_layers'):
                # dense layer
                if is_train:
                    output = tf.nn.dropout(output, 0.6, name='dropout')
                output = tf.layers.flatten(output, name='flatten')

            with tf.name_scope('output'):
                if self.is_train:
                    output = tf.layers.dense(output, classes)
                else:
                    output = tf.layers.dense(
                        output, classes, activation='softmax'
                    )

            return output

        return model

    def train_model(
        self, input, labels, epochs, batch_size,
        activation='softmax', save_path='./logs/checkpoints/checkpoint'
    ):
        if not self.is_train:
            logging.warning(
                'Model mode is not set for training (self.is_train=Flase).'
            )
            return
        if activation == 'softmax':
            model_output = tf.nn.softmax(self.model())
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=model_output, labels=labels
            )
        elif activation == 'sigmoid':
            model_output = tf.nn.sigmoid(self.model())

        global_step = tf.Variable(0, dtype=tf.int16, trainable=False)
        y = tf.placeholder(
            tf.float32, shape=[None, self.classes], name='labels'
        )
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=model_output, labels=y
            )
        )
        tf.summary.histogram('output', model_output)
        tf.summary.scalar('loss', loss)

        start_lr = 0.01
        lr = tf.train.exponential_decay(
            start_lr, global_step, decay_steps=epochs, decay_rate=0.5
        )
        tf.summary.scalar('learning_rate', lr)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            loss, global_step=global_step
        )
        summary_op = tf.summary.merge_all()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {
            self.input: input,
            y: labels
        }

        writer = tf.summary.FileWriter(
            './logs/{}'.format(self.timestamp), sess.graph
        )
        saver = tf.train.Saver(filename=save_path+'_'+self.timestamp+'.ckpt')

        for step in range(epochs):
            logging.info('epoch {}'.format(step))
            _, results = sess.run([train_op, summary_op], feed_dict=feed_dict)
            writer.add_summary(results, global_step=step)
            writer.flush()
            if step % 10 == 0:
                logging.info('Saving model...')
                saver.save(sess, os.path.dirname(save_path))
                logging.info('Model saved to {}.'.format(save_path))

    def evaluate_model(self):
        pass
