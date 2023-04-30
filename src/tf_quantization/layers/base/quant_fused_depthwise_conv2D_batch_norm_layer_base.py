import tensorflow as tf
from keras import initializers, regularizers, constraints, backend
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras import quantizers


class QuantFusedDepthwiseConv2DBatchNormalizationLayerBase(keras.layers.DepthwiseConv2D):

    def __init__(self, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, use_bias,
                 kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint,
                 bias_constraint, axis, momentum, epsilon, center, scale, beta_initializer,
                 gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer,
                 gamma_regularizer, beta_constraint, gamma_constraint, quantize, quantize_num_bits_weight,
                 per_channel, symmetric, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

        # TODO: I currently do not support more that 1 groups
        # BatchNormalization params
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer
        )
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.quantize = quantize
        self.quantize_num_bits_weight = quantize_num_bits_weight

        # Added param that allows batch norm freezing at the end of training
        self.frozen_bn = False

        # Quantization params
        self.per_channel = per_channel
        self.symmetric = symmetric

        self._quantizer_weights = None
        if quantize:
            self.weights_quantizer = quantizers.LastValueQuantizer(
                num_bits=quantize_num_bits_weight,
                per_axis=self.per_channel,
                symmetric=self.symmetric,
                narrow_range=True
            )
        else:
            self.weights_quantizer = None

    def get_config(self):
        base_config = super().get_config()
        config = {
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "moving_mean_initializer": initializers.serialize(
                self.moving_mean_initializer
            ),
            "moving_variance_initializer": initializers.serialize(
                self.moving_variance_initializer
            ),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
            "quantize": self.quantize,
            "quantize_num_bits_weight": self.quantize_num_bits_weight,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric
        }
        return dict(list(base_config.items()) + list(config.items()))

    def _get_folded_weights(self, std_dev, depthwise_kernel):
        gamma = tf.reshape(self.gamma, (1, 1, self.gamma.shape[0], 1))
        std_dev = tf.reshape(std_dev, (1, 1, std_dev.shape[0], 1))
        return (gamma / std_dev) * depthwise_kernel

    def _add_folded_bias(self, outputs, bias, mean, std_dev):
        bias = (bias - mean) * (
                self.gamma / std_dev) + self.beta
        return tf.nn.bias_add(
            outputs, bias, data_format=self._tf_data_format
        )

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        def calculate_update_delta():
            decay = tf.convert_to_tensor(1.0 - momentum, name="decay")
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = tf.where(
                    inputs_size > 0,
                    update_delta,
                    backend.zeros_like(update_delta),
                )
            return update_delta

        with backend.name_scope("AssignMovingAvg") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign_sub(calculate_update_delta(), name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign_sub(
                        variable, calculate_update_delta(), name=scope
                    )

    def _assign_new_value(self, variable, value):
        with backend.name_scope("AssignNewValue") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign(variable, value, name=scope)

    def _build_quantizer_weights(self):
        if self.weights_quantizer is not None:
            channellast_kernel = tf.transpose(self.depthwise_kernel, [0, 1, 3, 2])
            self._quantizer_weights = self.weights_quantizer.build(channellast_kernel.shape, "weights", self)

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    def freeze_bn(self):
        """
        Freezes BatchNorm in the layer (moving mean and variance won't be updated anymore)
        and training will use moving mean and variance instead of batch statistics

        Graph will be same as inference graph
        """
        self.frozen_bn = True

    def is_frozen(self):
        return self.frozen_bn