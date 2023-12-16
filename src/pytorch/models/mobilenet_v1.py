# Author: Jan Klhufek (iklhufek@fit.vut.cz)

import os
import torch
import torch.nn as nn
from torch.quantization import QConfig, default_per_channel_weight_observer, MinMaxObserver
import inspect
from typing import Callable, Optional, Dict, Any


def batch_norm_conv(in_channels: int, out_channels: int, stride: int, act_function: Optional[Callable[..., nn.Module]] = nn.ReLU) -> nn.Sequential:
    """
    Creates a convolution block with batch normalization and an activation function.

    Args:
        in_channels (int): Number of input channels for the convolution layer.
        out_channels (int): Number of output channels for the convolution layer.
        stride (int): Stride size for the convolution layer.
        act_function (Optional[Callable[..., nn.Module]]): Activation function to be used after convolution. Defaults to nn.ReLU.

    Returns:
        nn.Sequential: A sequntial container block of the batch normalized convolution.
    """
    inplace_supported = 'inplace' in inspect.signature(act_function).parameters
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        act_function(inplace=True) if inplace_supported else act_function(),
    )


def depthwise_separable_conv(in_channels: int, out_channels: int, stride: int, act_function: Optional[Callable[..., nn.Module]] = nn.ReLU) -> nn.Sequential:
    """
    Creates a depthwise separable convolution block.

    Args:
        in_channels (int): Number of input channels for the depthwise convolution.
        out_channels (int): Number of output channels for the pointwise convolution.
        stride (int): Stride for the depthwise convolution.
        act_function (Optional[Callable[..., nn.Module]]): Activation function to be used after each convolution. Defaults to nn.ReLU.

    Returns:
        nn.Sequential: A sequntial container block of the depthwise separable convolution block.
    """
    inplace_supported = 'inplace' in inspect.signature(act_function).parameters
    return nn.Sequential(
        # Depthwise convolution
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(num_features=in_channels),
        act_function(inplace=True) if inplace_supported else act_function(),

        # Pointwise convolution
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        act_function(inplace=True) if inplace_supported else act_function(),
    )


class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3, width_mult: float = 1.0, act_function: Optional[Callable[..., nn.Module]] = nn.ReLU, qat: bool = False, symmetric_quant: bool = False, per_channel_quant: bool = False, quant_config: Optional[Dict[int, Dict[str, int]]] = None, **kwargs) -> None:
        """
        Initializes a MobileNetV1 model.

        Args:
            num_classes (int): Number of classes for the classification task. Defaults to 1000.
            in_channels (int): Number of input channels. Defaults to 3.
            width_mult (float): Width multiplier to thin the network. Defaults to 1.0.
            act_function (Optional[Callable[..., nn.Module]]): Activation function to be used. Defaults to nn.ReLU.
            qat (bool): If True, enables Quantization-Aware Training. Defaults to False.
            symmetric_quant (bool): If True, uses symmetric quantization. Defaults to False.
            per_channel_quant (bool): If True, uses per-channel quantization. Defaults to False.
            quant_config (Optional[Dict[int, Dict[str, int]]]): Configuration for layer-specific quantization parameters.
                Each key is an index of a layer (important is the order, not key names), and each value is a dictionary with
                "Inputs" and "Weights" as keys representing the bitwidth for activations and weights, respectively.
                Example: {0: {"Inputs": 8, "Weights": 8}} sets 8-bit quantization for both activations and weights of the first layer.
                If None, default uniform 8-bit bitwidths are used. Defaults to None.
        """
        super().__init__()
        self._num_classes = num_classes
        self._width_mult = width_mult
        self._qat = qat
        self._symmetric_quant = symmetric_quant
        self._per_channel_quant = per_channel_quant

        # Prepare model for qat mode
        if self._qat:
            assert act_function == nn.ReLU, "QAT requires the activation function to be nn.ReLU for correctly fusing the layers (other fusers not yet implemented)"
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()

        self.model = nn.Sequential(
            # Initial convolutional layer followed by batch normalization and activation function
            batch_norm_conv(in_channels=in_channels, out_channels=self._adjust_channels(32), stride=2, act_function=act_function),

            # 13 DW separable convolution layer (including batch normalization and activation function)
            depthwise_separable_conv(self._adjust_channels(32), self._adjust_channels(64), stride=1, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(64), self._adjust_channels(128), stride=2, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(128), self._adjust_channels(128), stride=1, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(128), self._adjust_channels(256), stride=2, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(256), self._adjust_channels(256), stride=1, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(256), self._adjust_channels(512), stride=2, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(512), self._adjust_channels(512), stride=1, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(512), self._adjust_channels(512), stride=1, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(512), self._adjust_channels(512), stride=1, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(512), self._adjust_channels(512), stride=1, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(512), self._adjust_channels(512), stride=1, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(512), self._adjust_channels(1024), stride=2, act_function=act_function),
            depthwise_separable_conv(self._adjust_channels(1024), self._adjust_channels(1024), stride=1, act_function=act_function),

            # Setting global average pooling to output a feature map of size 1x1 for each channel (to reduce spatial dimensions to a single value)
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.classifier = nn.Linear(in_features=self._adjust_channels(1024), out_features=self._num_classes)
        self.initialize_weights()

        # Set qconfig once model is defined
        if self._qat:
            self._quant_config = quant_config if quant_config is not None else {}
            self._set_qat_config()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MobileNetV1 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        if self._qat:
            x = self.quant(x)
        x = self.model(x)
        x = x.flatten(start_dim=1)  # or also x.view(-1, 1024)
        x = self.classifier(x)
        if self._qat:
            x = self.dequant(x)
        return x

    def _adjust_channels(self, channels: int) -> int:
        """
        Adjusts the number of channels based on the width multiplier.

        Args:
            channels (int): Original number of channels.

        Returns:
            int: Adjusted number of channels.
        """
        return max(int(channels * self._width_mult), 1)

    def initialize_weights(self, only_fc: bool = False) -> None:
        """
        Initializes weights of the model's layers. Applies specific initializations for
        Conv2d, BatchNorm2d, and Linear layers. Skips Conv2d and BatchNorm2d if `only_fc` is True.

        Args:
            only_fc (bool): If True, only initializes weights for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not only_fc:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) and not only_fc:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

    def _set_qat_config(self) -> None:
        """
        Sets the Quantization-Aware Training (QAT) configuration (`qconfig`)
        for the model based on the specified quantization settings.
        """
        # Set default config for the whole model
        self.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

        # Determine the observer for activations
        activation_observer = MinMaxObserver
        # Determine the observer for weights
        weight_observer = default_per_channel_weight_observer if self._per_channel_quant else MinMaxObserver
        if self._per_channel_quant:  # Set the appropriate qscheme for per-channel quantization
            weight_qscheme = torch.per_channel_symmetric if self._symmetric_quant else torch.per_channel_affine
        else:  # Set the appropriate qscheme for per-tensor quantization
            weight_qscheme = torch.per_tensor_symmetric if self._symmetric_quant else torch.per_tensor_affine

        # Define qconfig for individual quantizable layers based on the configuration settings
        i = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                # Fetch the quantization config for the current layer
                config = self._quant_config.get(i, {"Inputs": 8, "Weights": 8})  # Set default bw to 8 if not specified

                activation_bw = config["Inputs"]
                weight_bw = config["Weights"]
                assert 2 <= activation_bw <= 8, "Activation bit-width must be between 2 and 8 bits"
                assert 2 <= weight_bw <= 8, "Weight bit-width must be between 2 and 8 bits"

                custom_qconfig = QConfig(
                    activation=activation_observer.with_args(
                        dtype=torch.quint8,
                        qscheme=torch.per_tensor_affine,
                        quant_min=0,
                        quant_max=2**activation_bw - 1
                    ),
                    weight=weight_observer.with_args(
                        dtype=torch.qint8,
                        qscheme=weight_qscheme,
                        quant_min=-(2**(weight_bw-1)),
                        quant_max=2**(weight_bw-1)-1
                    )
                )
                m.qconfig = custom_qconfig
                i += 1

    def prepare_for_qat(self) -> None:
        """
        Prepares the model for Quantization-Aware Training (QAT) and fuses model layers.
        """
        # Apply layer fusion and prepare the model for QAT
        self.eval()
        self.fuse_model()
        self.train()
        torch.quantization.prepare_qat(self, inplace=True)

    def fuse_model(self) -> None:
        """
        Fuses layers in the model to optimize it for quantization.

        Convolutions followed by BatchNorm and Activation (ReLU supported)
        are fused into a single layer to improve the performance and
        accuracy of the quantized model.
        """
        fuse_modules = torch.ao.quantization.fuse_modules_qat

        for m in self.model:
            if isinstance(m, nn.Sequential):
                fuse_modules(m, ['0', '1', '2'], inplace=True)  # Initial batch norm convolutiona block or pointwise convolution
                if len(m) > 3:  # This checks for the separable depthwise convolutional blocks (3 depthwise + 3 pointwise layers)
                    fuse_modules(m, ['3', '4', '5'], inplace=True)  # Pointwise

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def width_mult(self) -> float:
        return self._width_mult

    @property
    def qat(self) -> bool:
        return self._qat

    @property
    def per_channel_quant(self) -> bool:
        return self._per_channel_quant

    @property
    def symmetric_quant(self) -> bool:
        return self._symmetric_quant


def align_state_dict(model_state_dict: Dict[str, torch.Tensor], chkpt_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Aligns the checkpoint state dictionary with the model's state dictionary
    based on the structure of the keys.

    Args:
        model_state_dict (Dict[str, torch.Tensor]): State dictionary of the MobileNetV1 model.
        chkpt_state_dict (Dict[str, torch.Tensor]): State dictionary from the checkpoint.

    Returns:
        Dict[str, torch.Tensor]: The aligned state dictionary that can be loaded into the model.
    """
    chkpt = {k.replace('module.', ''): v for k, v in chkpt_state_dict.items()}

    # Iterate over model state dict keys and align with checkpoint keys
    aligned_state_dict = {}
    chkpt_keys = iter(chkpt.keys())
    current_chkpt_key = next(chkpt_keys, None)

    for key in model_state_dict.keys():
        # If keys match, use the value from the checkpoint
        if key.split('.')[-1] == current_chkpt_key.split('.')[-1]:
            aligned_state_dict[key] = chkpt[current_chkpt_key]
            current_chkpt_key = next(chkpt_keys, None)  # Move to the next key in the checkpoint
        # If keys don't match, use the value from the model
        # (for example if tracking the number of batches by batch norm layers is missing)
        else:
            aligned_state_dict[key] = model_state_dict[key]
    return aligned_state_dict


def load_quantized_model(model: MobileNetV1, checkpoint: Dict[str, Any], quantized: bool) -> MobileNetV1:
    """
    Loads a quantized model from a checkpoint.

    This function either loads the state dictionary from the last Quantization-Aware Training (QAT) checkpoint
    or loads the state of already converted (post training) quantized model.

    Args:
        model (MobileNetV1): The MobileNetV1 model to load the checkpoint into.
        checkpoint (Dict[str, Any]): The checkpoint containing the state dictionary and other information.
        quantized (bool): Flag indicating whether the model is quantized.

    Returns:
        MobileNetV1: The model loaded with the checkpoint.
    """
    # Prepare model for QAT and fuse layers
    if quantized:
        model.prepare_for_qat()
        model.to("cpu")  # NOTE: Quantization operations in PyTorch are optimized to for CPU backend inference (i.e. utilization of vectorization, etc.).
        model.eval()
        torch.quantization.convert(model, inplace=True)
    else:
        model.prepare_for_qat()
        model.eval()

    # Load the state dictionary
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_model(model: MobileNetV1, checkpoint) -> MobileNetV1:
    """
    Loads a model from a checkpoint with potential adjustments for the number of classes.

    This function aligns and loads the state dictionary from the checkpoint.
    If the number of classes in the checkpoint differs from the model, it adjusts the final linear layer accordingly.

    Args:
        model (MobileNetV1): The MobileNetV1 model to load the checkpoint into.
        checkpoint (Dict[str, Any]): The checkpoint containing the state dictionary and other information.

    Returns:
        MobileNetV1: The model loaded with the checkpoint.
    """
    chkpt_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    # Check the shape of the weights tensor of the last (linear) layer of checkpoint to match the number of model's classes (to allow loading the rest of the pretrained layers weights into the model and then adjust the linear layer)
    chkpt_fc_classes = (list(chkpt_state_dict.values())[-2]).shape[0]
    if model.num_classes == chkpt_fc_classes:
        state_dict = align_state_dict(model.state_dict(), chkpt_state_dict)
        print("Loading state dict from checkpoint..")
        model.load_state_dict(state_dict, strict=False)
    else:
        model.classifier = nn.Linear(model._adjust_channels(1024), chkpt_fc_classes)
        model.initialize_weights(only_fc=True)
        state_dict = align_state_dict(model.state_dict(), chkpt_state_dict)
        print("Loading state dict from checkpoint..")
        model.load_state_dict(state_dict, strict=False)
        # Adjust the number of classes desired for classification
        model.classifier = nn.Linear(model._adjust_channels(1024), model.num_classes)
        model.initialize_weights(only_fc=True)
        # Set qconfig again since last layer has been changed
        if model._qat:
            model._set_qat_config()
    return model


def mobilenetv1(num_classes: int = 1000, pretrained: bool = False, checkpoint_path: str = "", half_tensor: bool = False, load_quantized: bool = False, **kwargs) -> MobileNetV1:
    """
    Factory function for MobileNetV1.

    Args:
        num_classes (int): Number of classes for the classification task. Defaults to 1000.
        pretrained (bool): If True, loads pretrained weights. NOTE: Pretrained weights must
        be provided via a file path as MobileNetV1 is not offered pretrained via torchvision. Defaults to False.
        checkpoint_path (str): Path to the pretrained model weights. Required if pretrained is True. Defaults to "".
        half_tensor (bool): Specify whether to convert model parameters to half precision (FP16). Used mainly for faster inference. Defaults to False.
        load_quantized (bool): If True, premuses the checkpoint is already converted into quantized form. Defaults to False.
        **kwargs: Additional keyword arguments passed to the MobileNetV1 constructor.

    Returns:
        MobileNetV1: An instance of the MobileNetV1 model.
    """
    print("Creating custom MobileNetV1 model and initiliazing weights..")
    model = MobileNetV1(num_classes=num_classes, **kwargs)
    assert not (model.qat and half_tensor), "qat and half_tensor cannot be used simultaneously."
    if pretrained and checkpoint_path != "":
        assert os.path.exists(checkpoint_path), f"Pretrained model path '{checkpoint_path}' not found."
        checkpoint = torch.load(checkpoint_path)
        if model.qat and load_quantized:
            print("Loading quantized model\n")
            model = load_quantized_model(model, checkpoint, quantized=True)
        else:
            resume_qat = any('quant' in key for key in checkpoint['state_dict'].keys())
            if model.qat and resume_qat:  # check if resuming or starting training
                print("Loading QAT state model\n")
                model = load_quantized_model(model, checkpoint, quantized=False)
            else:
                assert not resume_qat, "Error while trying to load quantized state dict into non-qat ready instantiated model! Check your run settings."
                model = load_model(model, checkpoint)
                if model.qat:
                    print("Preparing model for QAT\n")
                    model.prepare_for_qat()
    else:
        if pretrained and checkpoint_path == "":
            print("You chose the `pretrained` option, but NOTE that MobileNetV1 is not provided pretrained within "
                  "the torchvision module, so just randomly initialized custom module is created instead.")
        if model.qat:
            print("Preparing model for QAT\n")
            model.prepare_for_qat()

    model = model.half() if half_tensor else model
    if half_tensor:
        print("Using half precision for model tensors.")
    return model
