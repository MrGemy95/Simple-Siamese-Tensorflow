from siamese_model import siamese_model
import tensorflow as tf
import os
from data_generator import DataLoader
import argparse


def test(args):
    batch_size = 16
    shape = (200, 100, 3)
    loader = DataLoader(args.path, 150, shape[:2])
    exp_path = "./"

    model = siamese_model(shape)
    optim = tf.keras.optimizers.Adam(lr=0.0001)

    loss = 'binary_crossentropy'
    metrics = ['binary_accuracy', 'acc']

    model.compile(loss=loss,
                  optimizer=optim,
                  metrics=metrics)

    model.load_weights(os.path.join(exp_path, args.exp_name, "ckpt"))
    print("Model Loaded Successifully")
    model.summary()
    #
    # tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_path, args.exp_name, "logs"),
    #                                     histogram_freq=0,
    #                                     write_graph=True,
    #                                     write_images=True)
    #
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(exp_path, args.exp_name, "ckpt"),
    #                                                 monitor='val_acc',
    #                                                 verbose=1,
    #                                                 save_weights_only=True,
    #                                                 save_best_only=True,
    #                                                 mode='max')
    # callbacks_list = [checkpoint, tf.keras.callbacks.TerminateOnNaN(), tb]

    history = model.evaluate(loader.generate_test(batch_size),
                             steps=loader.test_size // batch_size
                             )


if __name__ == '__main__':
    pass
