from siamese_model import siamese_model
import tensorflow as tf
import os
from data_generator import DataLoader
import argparse
import imageio
import skimage


def test(args):
    shape = (200, 100, 3)
    exp_path = "./"
    img1 = imageio.imread(args.img1)
    img1 = skimage.transform.resize(img1, shape, preserve_range=True)
    img1 = tf.keras.applications.resnet50.preprocess_input(img1)

    img2 = imageio.imread(args.img2)
    img2 = skimage.transform.resize(img2, shape, preserve_range=True)
    img2 = tf.keras.applications.resnet50.preprocess_input(img2)

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

    history = model.predict([[img1], [img2]], batch_size=1)
    print(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", help="path to img", default="../")
    parser.add_argument("--img2", help="path to img", default="../")
    parser.add_argument("--exp_name", help="experiment name", default="exp")
    args = parser.parse_args()

    test(args)
