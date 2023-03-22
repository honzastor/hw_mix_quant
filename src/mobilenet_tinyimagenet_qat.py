import argparse
from abc import ABC
from datetime import datetime

from tensorflow import keras
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from tf_quantization.layers.quant_conv2D_batch_layer import QuantConv2DBatchLayer
from tf_quantization.layers.quant_depthwise_conv2d_bn_layer import QuantDepthwiseConv2DBatchNormalizationLayer
from tf_quantization.quantize_model import quantize_model
from datasets import tinyimagenet
import os
import numpy as np


def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold=0,
                           total_steps=0,
                           target_lr=1e-3):
    # From https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    global_step = tf.dtypes.cast(global_step, dtype=tf.float32)
    learning_rate = 0.5 * target_lr * (1 + tf.cos(
        tf.constant(np.pi) * (global_step - warmup_steps - hold) / float(
            total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule, ABC):
    # From https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
    def __init__(self, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(global_step=step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    target_lr=self.target_lr,
                                    hold=self.hold)

        return tf.where(
            step > self.total_steps, 0.0, lr, name="learning_rate"
        )

    def get_config(self):
        config = {
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'hold': self.hold
        }
        return config


def main(*, q_aware_model, epochs, bn_freeze=10e1000, batch_size=64, learning_rate=0.001, warmup=0.0,
         checkpoints_dir=None, logs_dir=None,
         cache_dataset=True, from_checkpoint=None, verbose=False):
    if verbose:
        print("Used configuration:")
        print(f'Number of epochs: {epochs}')
        print(f'Batch Size: {batch_size}')
        print(f'Learning rate: {learning_rate}')
        print(f'Warmup: {warmup}')
        print(f'Checkpoints directory: {checkpoints_dir}')
        print(f'Logs directory: {logs_dir}')
        print(f'Cache training dataset: {cache_dataset}')
        if from_checkpoint is not None:
            print(f'From checkpoint: {from_checkpoint}')

    if args.weight_bits < 2 or args.weight_bits > 8:
        raise ValueError("Weight bits must be in <2,8> interval.")

    if args.warmup < 0 or args.warmup > 1:
        raise ValueError("Warmup % must be in <0,1> interval.")

    # Load dataset
    tr_ds = tinyimagenet.get_tinyimagenet_dataset(split="train")
    tr_ds = tr_ds.map(tinyimagenet.get_preprocess_image_fn(image_size=(224, 224)))

    if cache_dataset:
        tr_ds = tr_ds.cache()

    train_ds = tr_ds.map(lambda data: (data['image'], data['label']))
    train_ds = train_ds.shuffle(30000).batch(batch_size)

    ds = tinyimagenet.get_tinyimagenet_dataset(split="val")
    ds = ds.map(tinyimagenet.get_preprocess_image_fn(image_size=(224, 224)))

    if cache_dataset:
        ds = ds.cache()

    test_ds = ds.map(lambda data: (data['image'], data['label'])).batch(batch_size)

    if from_checkpoint is not None:
        q_aware_model.load_weights(from_checkpoint)

    total_steps = len(train_ds) * epochs
    warmup_steps = int(warmup * total_steps)  # do not use warmup, only cosine decay

    schedule = WarmUpCosineDecay(target_lr=learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,
                                 hold=warmup_steps)

    if not from_checkpoint:
        q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])

        # Train activation moving averages
        q_aware_model.fit(train_ds, epochs=3)

    q_aware_model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=schedule, momentum=0.9),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

    if verbose:
        qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
        print(f'Top-1 accuracy before QAT (quantized float): {qa_acc * 100:.2f}%')

    # Define checkpoint callback for saving model weights after each epoch
    callbacks = []
    if checkpoints_dir is not None:
        checkpoints_dir = os.path.abspath(checkpoints_dir)
        checkpoints_dir = os.path.join(checkpoints_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(checkpoints_dir)

        checkpoint_filepath = checkpoints_dir + '/weights-{epoch:03d}-{val_accuracy:.4f}.hdf5'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode="max"
        )
        callbacks.append(model_checkpoint_callback)

    # Define the Keras TensorBoard callback.
    if logs_dir is not None:
        logs_dir = os.path.abspath(logs_dir)
        logs_dir = os.path.join(logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(logs_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)
        callbacks.append(tensorboard_callback)

    not_frozen_epochs = min(epochs, bn_freeze)

    # Train with not frozen batch norms
    q_aware_model.fit(train_ds, epochs=not_frozen_epochs, validation_data=test_ds,
                      callbacks=callbacks,
                      initial_epoch=0)

    if epochs > bn_freeze:
        # Train with bn frozen
        _freeze_bn_in_model(q_aware_model)
        q_aware_model.fit(train_ds, epochs=epochs, validation_data=test_ds,
                          callbacks=callbacks,
                          initial_epoch=bn_freeze)

    qa_loss, qa_acc = q_aware_model.evaluate(test_ds)
    if verbose:
        print(f'Top-1 accuracy after (quantized float): {qa_acc * 100:.2f}%')
    return qa_acc


def _freeze_bn_in_model(model):
    for layer in model.layers:
        if (isinstance(layer, QuantConv2DBatchLayer) or
                isinstance(layer, QuantDepthwiseConv2DBatchNormalizationLayer)):
            layer.freeze_bn()
        if isinstance(layer, QuantizeWrapper):
            if isinstance(layer.layer, keras.layers.BatchNormalization):
                layer.trainable = False
                layer.training = False


if __name__ == "__main__":
    tf.random.set_seed(30082000)  # Set random seed to have reproducible results

    # Script arguments
    parser = argparse.ArgumentParser(
        prog='mobilenet_tinyimagenet_qat',
        description='Quantize mobilenet',
        epilog='')

    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('--bn-freeze', default=25, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)

    parser.add_argument('--weight-bits', '--wb', default=8, type=int)

    parser.add_argument('--learning-rate', '--lr', default=0.01, type=float)
    parser.add_argument('--warmup', default=0.05, type=float)

    parser.add_argument("--logs-dir", default="logs/tinyimagenet/mobilenet/8bit")
    parser.add_argument("--checkpoints-dir", default="checkpoints/tinyimagenet/mobilenet/8bit")

    parser.add_argument('--from-checkpoint', default=None, type=str)
    parser.add_argument('--start-epoch', default=0, type=int)

    parser.add_argument('-v', '--verbose', default=False, action='store_true')  # on/off flag
    parser.add_argument('--cache', default=False, action='store_true')  # on/off flag
    parser.add_argument('--mobilenet-path', default="mobilenet_tinyimagenet.keras", type=str)

    args = parser.parse_args()
    model = keras.models.load_model(args.mobilenet_path)

    quant_layer_conf = {"weight_bits": args.weight_bits, "activation_bits": 8}
    q_aware_model = quantize_model(model, [quant_layer_conf for _ in range(37)])

    main(
        q_aware_model=q_aware_model,
        epochs=args.epochs,
        bn_freeze=args.bn_freeze,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup=args.warmup,
        checkpoints_dir=args.checkpoints_dir,
        logs_dir=args.logs_dir,
        cache_dataset=args.cache,
        from_checkpoint=args.from_checkpoint,
        verbose=args.verbose
    )