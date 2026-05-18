"""Auto-split from legacy monolithic script."""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionWithWeights(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1), initializer="random_normal"
        )

    def call(self, inputs):
        scores = tf.matmul(inputs, self.W)
        weights = tf.nn.softmax(scores, axis=1)
        output = tf.reduce_sum(inputs * weights, axis=1)
        return (output, weights)
