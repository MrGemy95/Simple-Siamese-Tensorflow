import tensorflow as tf

tf.keras.applications.xception.Xception()


def siamese_model(shape):
    img_a = tf.keras.layers.Input(shape=shape)
    img_b = tf.keras.layers.Input(shape=shape)
    backbone = tf.keras.Sequential()
    resnet = tf.keras.applications.resnet50.ResNet50(weights=None, input_shape=(221, 221, 3),
                                                     include_top=False, pooling='avg')

    # for layer in resnet.layers[0:-20]:
    #     layer.trainable = False

    backbone.add(resnet)

    features_a = backbone(img_a)
    features_b = backbone(img_b)

    merged = tf.keras.backend.concatenate([features_a, features_b])
    fc1 = tf.keras.layers.Dense(128, activation='relu', name='fc_1')(merged)
    fc1 = tf.keras.layers.Dropout(0.7)(fc1)
    fc2 = tf.keras.layers.Dense(32, activation='relu', name='fc_2')(fc1)
    fc2 = tf.keras.layers.Dropout(0.3)(fc2)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='fc_3')(fc2)

    siamese = tf.keras.Model([img_a, img_b], out)
    return siamese
