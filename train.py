from siamese_model import siamese_model
import tensorflow as tf
import os
from data_generator import DataLoader
import argparse


def train(args):
    batch_size = 16
    shape = (200, 100, 3)
    loader = DataLoader(args.path, args.samples, shape[:2], augment=args.aug)
    exp_path = "./"

    model = siamese_model(shape)
    optim = tf.keras.optimizers.Adam(lr=0.0001)

    loss = 'binary_crossentropy'
    metrics = ['binary_accuracy', 'acc']

    model.compile(loss=loss,
                  optimizer=optim,
                  metrics=metrics)

    model.summary()

    tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_path, args.exp_name, "logs"),
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(exp_path, args.exp_name, "ckpt"),
                                                    monitor='val_acc',
                                                    verbose=1,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    mode='max')
    callbacks_list = [checkpoint, tf.keras.callbacks.TerminateOnNaN(), tb]

    history = model.fit_generator(loader.generate_epoch_train(batch_size),
                                  validation_data=loader.generate_epoch_val(batch_size),
                                  validation_steps=loader.val_size // batch_size,
                                  steps_per_epoch=loader.train_size // batch_size,
                                  epochs=700,
                                  callbacks=callbacks_list)

    acc = history.history['val_binary_accuracy']
    tacc = history.history['binary_accuracy']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to dataset", default="../")
    parser.add_argument("--exp_name", help="experiment name", default="exp")
    parser.add_argument("--samples", type=int, help="samples", default=100)
    parser.add_argument("--aug", help="data augmentataion", action="store_true", default=False)
    args = parser.parse_args()

    train(args)
