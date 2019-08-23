import numpy as np
import h5py
import os
import random
import sys
from random import shuffle
import glob
from scipy import misc
import matplotlib.pyplot as plt
import skimage
import imageio
import tensorflow as tf

random.seed(0)


class DataLoader():
    def __init__(self, path, samples, img_size, test=False):
        self.path = path
        self.img_size = img_size
        self.samples = samples
        if test:
            self.test_data = self.prepare_test_data(path)
            self.test_size = len(self.test_data)

        else:

            self.train_data, self.val_data = self.prepare_train_data(path)
            self.train_size = len(self.train_data)
            self.val_size = len(self.val_data)

    def prepare_train_data(self, path, train_ratio=.8):
        all_dirs = sorted(os.listdir(path=os.path.join(path, "bbox_train")))
        print(all_dirs)
        shuffle(all_dirs)
        print(all_dirs)
        train_idx = int(train_ratio * len(all_dirs))
        train_dirs = all_dirs[:train_idx]
        val_dirs = all_dirs[train_idx:]
        train_data = self.load_data(os.path.join(path, "bbox_train"), train_dirs, self.samples, self.img_size)
        val_data = self.load_data(os.path.join(path, "bbox_train"), val_dirs, self.samples, self.img_size)
        return train_data, val_data

    def prepare_test_data(self, path):
        all_dirs = sorted(os.listdir(path=os.path.join(path, "bbox_test")))
        print(all_dirs)
        shuffle(all_dirs)
        print(all_dirs)
        test_data = self.load_data(os.path.join(path, "bbox_test"), all_dirs, self.samples, self.img_size)
        return test_data

    def get_pair_test(self, positive):

        if positive:
            idxs = np.random.choice(self.test_size, 1)
            sample = np.random.choice(self.samples, 2)
            imgs = self.test_data[idxs[0], sample]
        else:
            idxs = np.random.choice(self.test_size, 2)
            sample = np.random.choice(self.samples, 1)
            imgs = self.test_data[idxs, sample[0]]
        return imgs

    def get_pair(self, positive, split):
        data = self.train_data if split == "train" else self.val_data
        size = self.train_size if split == "train" else self.val_size
        if positive:
            idxs = np.random.choice(size, 1)
            sample = np.random.choice(self.samples, 2)
            imgs = data[idxs[0], sample]
        else:
            idxs = np.random.choice(size, 2)
            sample = np.random.choice(self.samples, 1)
            imgs = data[idxs, sample[0]]

        # temp=np.concatenate([imgs[0],imgs[1]],axis=1)
        # plt.imshow(temp)
        # plt.show()
        return imgs

    def generate_epoch_train(self, batch_size):
        while True:
            batch_images1 = []
            batch_images2 = []
            labels = []
            for i in range(batch_size // 2):
                imgs = self.get_pair(True, "train")
                batch_images1.append(imgs[0])
                batch_images2.append(imgs[1])
                labels.append([1., 0.])
                imgs = self.get_pair(False, "train")
                batch_images1.append(imgs[0])
                batch_images2.append(imgs[1])
                labels.append([0., 1.])
            batch_images1 = np.array(batch_images1)
            batch_images2 = np.array(batch_images2)
            labels = np.array(labels)

            yield [batch_images1, batch_images2], labels[:, 0]

    def generate_test(self, batch_size):
        while True:
            batch_images1 = []
            batch_images2 = []
            labels = []
            for i in range(batch_size // 2):
                imgs = self.get_pair_test(True)
                batch_images1.append(imgs[0])
                batch_images2.append(imgs[1])
                labels.append([1., 0.])
                imgs = self.get_pair_test(False)
                batch_images1.append(imgs[0])
                batch_images2.append(imgs[1])
                labels.append([0., 1.])
            batch_images1 = np.array(batch_images1)
            batch_images2 = np.array(batch_images2)
            labels = np.array(labels)

            yield [batch_images1, batch_images2], labels[:, 0]

    def generate_epoch_val(self, batch_size):
        while True:
            batch_images1 = []
            batch_images2 = []
            labels = []
            for i in range(batch_size // 2):
                imgs = self.get_pair(True, "train")
                batch_images1.append(imgs[0])
                batch_images2.append(imgs[1])
                labels.append([1., 0.])
                imgs = self.get_pair(False, "val")
                batch_images1.append(imgs[0])
                batch_images2.append(imgs[1])
                labels.append([0., 1.])
            batch_images1 = np.array(batch_images1)
            batch_images2 = np.array(batch_images2)
            labels = np.array(labels)
            yield [batch_images1, batch_images2], labels[:, 0]

    def load_data(self, path, dirs, samples=10, size=(256, 128)):
        print(len(dirs), samples, size[0], size[1], 3)
        data = np.zeros(shape=(len(dirs), samples, size[0], size[1], 3), dtype=np.float32)

        for i, dir in enumerate(dirs):
            files = [f for f in glob.glob(os.path.join(path, dir) + "/*.jpg")]
            print(dir)
            sample_files = random.choices(files, k=samples)
            for j, s in enumerate(sample_files):
                img = imageio.imread(s)

                img = skimage.transform.resize(img, size, preserve_range=True)
                # plt.imshow(img)
                # plt.show()
                # test = tf.keras.applications.resnet50.preprocess_input(img)
                data[i, j] = tf.keras.applications.resnet50.preprocess_input(img)
                # plt.imshow(data[i, j])
                # plt.show()
        return data


#
# def get_pair(path, set, ids, positive):
#     pair = []
#     pic_name = []
#     files = os.listdir('%s/%s' % (path, set))
#     if positive:
#         value = random.sample(ids, 1)
#         id = [str(value[0]), str(value[0])]
#     else:
#         id = random.sample(ids, 2)
#     id = [str(id[0]), str(id[1])]
#     for i in range(2):
#         # id_files = [f for f in files if (f[0:4] == ('%04d' % id[i]) or (f[0:2] == '-1' and id[i] == -1))]
#         id_files = [f for f in files if f.split('_')[0] == id[i]]
#         pic_name.append(random.sample(id_files, 1))
#     for pic in pic_name:
#         pair.append('%s/%s/' % (path, set) + pic[0])
#
#     return pair
#
#
# '''
# def get_num_id(path, set):
#     files = os.listdir('%s/%s' % (path, set))
#     files.sort()
#     return int(files[-1].split('_')[0]) + 1
# '''
#
#
# def get_id(path, set):
#     files = os.listdir('%s/%s' % (path, set))
#     IDs = []
#     for f in files:
#         IDs.append(f.split('_')[0])
#     IDs = list(set(IDs))
#     return IDs
#
#
# def read_data(path, set, ids, image_width, image_height, batch_size):
#     batch_images = []
#     labels = []
#     for i in range(batch_size // 2):
#         pairs = [get_pair(path, set, ids, True), get_pair(path, set, ids, False)]
#         for pair in pairs:
#             images = []
#             for p in pair:
#                 image = cv2.imread(p)
#                 image = cv2.resize(image, (image_width, image_height))
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 images.append(image)
#             batch_images.append(images)
#         labels.append([1., 0.])
#         labels.append([0., 1.])
#
#     '''
#     for pair in batch_images:
#         for p in pair:
#             cv2.imshow('img', p)
#             key = cv2.waitKey(0)
#             if key == 1048603:
#                 exit()
#     '''
#     return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)


# if __name__ == '__main__':
# prepare_data(sys.argv[1])
if __name__ == '__main__':
    loader = DataLoader("/home/gemy/work/freelancing/mars-motion-analysis-and-reidentification-set/", 10, (200, 100),
                        test=True)
    for i, j in enumerate(loader.generate_test(8)):
        print(i)
    for i, j in enumerate(loader.generate_epoch_val(8)):
        print(i)
