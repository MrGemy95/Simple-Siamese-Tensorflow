from siamese_model import siamese_model
import tensorflow as tf
import os
from data_generator import DataLoader


def train():
    shape = (200, 100, 3)
    loader = DataLoader("/home/gemy/work/freelancing/mars-motion-analysis-and-reidentification-set/", 100, shape[:2])
    exp_path = "./"

    model = siamese_model(shape)
    optim = tf.keras.optimizers.Adam(lr=0.0001)

    loss = 'binary_crossentropy'
    metrics = ['binary_accuracy', 'acc']

    model.compile(loss=loss,
                  optimizer=optim,
                  metrics=metrics)

    model.summary()

    tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_path, "logs"),
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(exp_path, "ckpt"),
                                                    monitor='val_acc',
                                                    verbose=1,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    mode='max')
    callbacks_list = [checkpoint, tf.keras.callbacks.TerminateOnNaN(), tb]

    history = model.fit_generator(loader.generate_epoch_train(),
                                  validation_data=loader.generate_epoch_val(),
                                  validation_steps=50,
                                  steps_per_epoch=100,
                                  epochs=2000,
                                  callbacks=callbacks_list)

    acc = history.history['val_binary_accuracy']
    tacc = history.history['binary_accuracy']


if __name__ == '__main__':
    train()
