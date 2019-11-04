import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import math
import cv2
import numpy as np
from albumentations import RandomBrightnessContrast, RandomRain, Compose, ShiftScaleRotate, RandomSunFlare, OneOf, Blur
from copy import deepcopy


def readTrafficSignsFinal(rootpath):
    """
    Reads the final test images
    Example : './GSTRB/Final_Training/
    :param rootpath: the root folder of images
    :return: images and labels derived from dataset
    """
    images = []
    labels = []
    prefix = rootpath + 'Images/'
    gtFile = open(prefix + 'GT-final_test.csv', 'r')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)
    for row in gtReader:
        images.append(plt.imread(prefix + row[0]))
        labels.append(int(row[7]))
    gtFile.close()
    return images, labels


def train_test_split(rootpath):
    """
    Reads training data and splits it for training and testing part
    Example: './GSTRB/Final_Training/Images'
    :param rootpath: the root folder where tracks are stored
    :return: splitted dataset with train and test images and labels
    """
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for c in range(43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = pd.read_csv(prefix + 'GT-' + format(c, '05d') + '.csv', delimiter=';')

        dict_fnames = dict(zip(gtFile.Filename, gtFile.ClassId))
        track_list = sorted({track.split('_')[0] for track in dict_fnames.keys()})
        train_dict = track_list[:math.floor(0.8 * len(track_list))]

        for filename in dict_fnames.keys():
            if filename.split('_')[0] in train_dict:
                train_X.append(plt.imread(prefix + filename))
                train_y.append(int(dict_fnames[filename]))
            else:
                test_X.append(plt.imread(prefix + filename))
                test_y.append(int(dict_fnames[filename]))
    zipped_train = list(zip(train_X, train_y))
    random.shuffle(zipped_train)
    zipped_test = list(zip(test_X, test_y))
    random.shuffle(zipped_test)
    train_X, train_y = list(zip(*zipped_train))
    test_X, test_y = list(zip(*zipped_test))
    return list(train_X), list(train_y), list(test_X), list(test_y)


def padding_and_resizing(dataset, size=30):
    """
    Makes padding for images to make them in square form and after that resize them
    :param dataset: array of images
    :param size: which image size will be after applying function
    :return: list of padded and resized images
    """
    new_arr = []
    for image in dataset:
        max_edge = max(image.shape)
        add_width = max_edge - image.shape[1]
        add_height = max_edge - image.shape[0]
        padded = cv2.copyMakeBorder(image, add_height, 0, 0, add_width, cv2.BORDER_REPLICATE)
        resized = cv2.resize(padded, (size, size))
        new_arr.append(resized)
    return new_arr


def normalize(dataset):
    """
    Normalize each image in range of [0..1]
    :param dataset: array of images
    :return: normalized list of images
    """
    test_normed = []
    for img in dataset:
        test_normed.append(np.ravel(img) / 255)
    return test_normed


def show_train_freq(train_y, title):
    """
    Helper function which draws frequency distribution of train labels of classes
    :param train_y: labels of train images
    :param title: name of file which will be saved
    """
    plt.figure(figsize=(8, 8))
    plt.hist(train_y, bins=43, rwidth=0.85)
    plt.savefig(title)
    plt.xlabel("Class IDs")
    plt.ylabel('Number of samples')
    plt.show()


def augment_and_show(aug, image, title):
    """
    Helper function which draws augmentation
    :param aug: type of augmentation
    :param image: array based image
    :param title: name of file which will be saved
    """
    aug_image = aug(image=image)['image']
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Normal')
    ax[0].imshow(image)
    ax[1].set_title(title)
    ax[1].imshow(aug_image)
    plt.tight_layout()
    plt.savefig(title + '.png')
    plt.show()


def composition(p=1):
    """
    :param p: probability of applying list of augmentations
    :return: composition of augmentation
    """
    rain = random.choice([None, "drizzle", "heavy"])
    return Compose([
        ShiftScaleRotate(),
        OneOf([
            Blur(blur_limit=2),
            RandomSunFlare(src_radius=20),
            RandomBrightnessContrast(),
            RandomRain(blur_value=2, drop_width=1, rain_type=rain)
        ], p=1)

    ], p=p)


def augmentation(train_X, train_y):
    """
    Function that makes augmentations
    :param train_X: array of training images
    :param train_y: array of training labels
    :return: augmented arrays of images and labels
    """
    median = 2000
    c_train_y = deepcopy(train_y)
    for i in range(43):
        inds = np.argwhere(np.array(c_train_y) == i)
        num_samples = len(inds)
        while num_samples < median:
            rand_img = train_X[random.choice(inds)[0]]
            aug = composition()
            aug_img = aug(image=rand_img)['image']
            train_X.append(aug_img)
            c_train_y.append(i)

            num_samples += 1

    new_dataset = list(zip(train_X, c_train_y))
    random.shuffle(new_dataset)
    newTrain_X, newTrain_y = list(zip(*new_dataset))
    return newTrain_X, newTrain_y


def show_padded_and_resized(image, size, title='Resized and Padded'):
    """
    Helper function which draws normal image and padded and resized image of the same picture
    :param image: array based image
    :param size: which image size will be after applying function
    :param title: name of file which will be saved
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Normal')
    ax[0].imshow(image[0])
    ax[1].set_title(title)
    ax[1].imshow(padding_and_resizing(image, size)[0])
    plt.tight_layout()
    plt.savefig(title + '.png')
    plt.show()


def show_wrong_predictions(y_pred, y_true, test_X, size):
    """
    Shows wrong predicted image and true image from predicted class
    :param y_pred: predicted labels
    :param y_true: true labels
    :param test_X: arrays of test images
    :param size:  which image size will be after applying function
    """
    wrong_pred = []
    for i, pred in enumerate(y_pred):
        if pred != y_true[i]:
            wrong_pred.append(i)
    for i in range(4):
        plt.figure(figsize=(8, 8))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_X[wrong_pred[i]].reshape(size, size, 3))
        ax[0].set_title(f'Predicted label: {y_pred[wrong_pred[i]]}, true: {y_true[wrong_pred[i]]}')
        ax[1].imshow(test_X[np.argwhere(y_true == y_pred[wrong_pred[i]])[0][0]].reshape(size, size, 3))
        ax[1].set_title(f'true example of class {y_pred[wrong_pred[i]]}')
        plt.tight_layout()
        plt.savefig(f"example_{i}.png")
        plt.show()
