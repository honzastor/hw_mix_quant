from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_model, \
    quantize_apply, quantize_scope

from tf_quantization.layers.approx.quant_conv2D_batch_layer import ApproximateQuantConv2DBatchLayer
from tf_quantization.layers.approx.quant_depthwise_conv2d_bn_layer import \
    ApproximateQuantDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.layers.quant_conv2D_batch_layer import QuantConv2DBatchLayer
from tf_quantization.layers.quant_depthwise_conv2d_bn_layer import QuantDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.quantize_registry import PerLayerNBitQuantizeRegistry
from tf_quantization.transforms.quantize_transforms import PerLayerQuantizeModelTransformer, \
    PerLayerQuantizeLayoutTransform


class PerLayerNBitQuantizeScheme(quantize_scheme.QuantizeScheme):
    """Scheme that allows usage of any layout transformer and registry"""

    def __init__(self, transformer_fn, registry_fn):
        self.transformer_fn = transformer_fn
        self.registry_fn = registry_fn

    def get_layout_transformer(self):
        return self.transformer_fn()

    def get_quantize_registry(self):
        return self.registry_fn()


def quantize_model(model, quantization_config, activation_quant_no_affect=False, approx=False, per_channel=True,
                   symmetric=True):
    """
    Quantize model with option to configure quantization for each layer
    :param model: Model for quantization
    :param quantization_config: List of quantization configs for each layer
    :param activation_quant_no_affect: Disable activation quantization
    :param approx: Use approx batch norm scheme or more accurate version
    :param per_channel: Use per-channel quantization for weights of convolution layers
    :param symmetric: Use symmetric or asymmetric quantization for weights of convolution layers
    :return: Quantization aware model
    """

    with quantize_scope({
        "ApproximateQuantConv2DBatchLayer": ApproximateQuantConv2DBatchLayer,
        "QuantConv2DBatchLayer": QuantConv2DBatchLayer,
        "QuantDepthwiseConv2DBatchNormalizationLayer": QuantDepthwiseConv2DBatchNormalizationLayer,
        "ApproximateQuantDepthwiseConv2DBatchNormalizationLayer": ApproximateQuantDepthwiseConv2DBatchNormalizationLayer,

    }):
        transformer = PerLayerQuantizeModelTransformer(model, quantization_config, {}, approx=approx,
                                                       symmetric=symmetric, per_channel=per_channel)

        if transformer.get_number_of_quantizable_layers() != len(quantization_config):
            raise ValueError(
                f'There is different number of quantizable layers in model than in quantization config. ' +
                f'({transformer.get_number_of_quantizable_layers()} needed)')

        layer_group_map = transformer.get_layers_quantize_group_map()

        layer_quantization_config_map = {}
        for layer in layer_group_map.keys():
            layer_quantization_config_map[layer] = quantization_config[layer_group_map[layer]]

        scheme = PerLayerNBitQuantizeScheme(
            transformer_fn=lambda: PerLayerQuantizeLayoutTransform(quantization_config, approx=approx,
                                                                   symmetric=symmetric, per_channel=per_channel),
            registry_fn=lambda: PerLayerNBitQuantizeRegistry(
                layer_quantization_config_map,
                activation_quant_no_affect=activation_quant_no_affect,
                symmetric=symmetric,
                per_channel=per_channel
            )
        )

        annotated_model = quantize_annotate_model(model)
        return quantize_apply(annotated_model, scheme=scheme)
