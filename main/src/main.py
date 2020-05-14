print('\n \n Importing...\n')
import os
import re
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import color, transform, restoration, io, feature, img_as_ubyte
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# !
# IMPLIMENT THAT FUCKIN CLASS YO!
# !

class ImagePipeline(object):

    def __init__(self, parent_dir):
        """
        Manages reading, transforming and saving images
        :param parent_dir: Name of the parent directory containing all the sub directories
        """
        # Define the Paths and Directories
        self.parent_dir = parent_dir
        self.cur_path = os.getcwd()

        # Sub directory variables that are filled in when read()
        self.raw_sub_dir_names = None
        self.sub_dirs = None
        self.label_map = None

        # Image variables that are filled in when read() and vectorize()
        self.img_lst2 = []
        self.img_names2 = []
        self.features = None
        self.labels = None

    def _make_label_map(self):
        """
        Get the sub directory names and map them to numeric values (labels)
        :return: A dictionary of dir names to numeric values
        """
        return {label: i for i, label in enumerate(self.raw_sub_dir_names)}

    def _path_relative_to_parent(self, some_dir):
        """
        Get the full path of a sub directory relative to the parent
        :param some_dir: The name of a sub directory
        :return: Return the full path relative to the parent
        """
        cur_path = os.getcwd()
        return os.path.join(cur_path, some_dir)

    def _make_new_dir(self, new_dir):
        """
        Make a new sub directory with fully defined path relative to the parent directory
        :param new_dir: The name of a new sub dir
        """
        # Make a new directory for the new transformed images
        new_dir = os.path.join(self.cur_path, new_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        else:
            raise Exception('Directory already exist, please check...')
            

    def _empty_variables(self):
        """
        Reset all the image related instance variables
        """
        self.img_lst2 = []
        self.img_names2 = []
        self.features = None
        self.labels = None

    @staticmethod
    def _accepted_dir_name(dir_name):
        """
        Return boolean of whether the directory is of the accepted name (i.e. no hidden files)
        :param dir_name: Name of the directory in question
        :return: True or False (if the directory is hidden or not)
        """
        if dir_name.startswith('.'):
            return False
        else:
            return True

    def _assign_sub_dirs(self, sub_dirs=['all']):
        """
        Assign the names (self.raw_sub_dir_names) and paths (self.sub_dirs) based on
        the sub dirs that are passed in, otherwise will just take everything in the
        parent dir
        :param sub_dirs: Tuple contain all the sub dir names, else default to all sub dirs
        """
        # Get the list of raw sub dir names
        # self.sub_dirs = os.listdir(self.parent_dir)
        # self.sub_dirs.sort()
        # self.sub_dirs = [os.path.join(self.parent_dir, sub_dir) for sub_dir in self.sub_dirs]

        if sub_dirs[0] == 'all':
            self.raw_sub_dir_names = os.listdir(self.parent_dir)
            self.raw_sub_dir_names.sort()
        else:
            self.raw_sub_dir_names = sub_dirs

        # Make label to map raw sub dir names to numeric values
        self.label_map = self._make_label_map()

        # Get the full path of the raw sub dirs
        filtered_sub_dirs = list(
            filter(self._accepted_dir_name, self.raw_sub_dir_names))
        # self.sub_dirs = map(self._path_relative_to_parent, filtered_sub_dir)
        for i, filtered_sub_dir in enumerate(filtered_sub_dirs):
            filtered_sub_dirs[i] = os.path.join(
                self.cur_path, self.parent_dir, filtered_sub_dir)
        self.sub_dirs = filtered_sub_dirs

    def read(self, sub_dirs=['all']):
        """
        Read images from each sub directories into a list of matrix (self.img_lst2)
        :param sub_dirs: Tuple contain all the sub dir names, else default to all sub dirs
        """
        # Empty the variables containing the image arrays and image names, features and labels
        self._empty_variables()

        # Assign the sub dir names based on what is passed in
        self._assign_sub_dirs(sub_dirs=sub_dirs)

        for sub_dir in self.sub_dirs:
            img_names = os.listdir(sub_dir)
            self.img_names2.append(img_names)

            img_lst = [io.imread(os.path.join(sub_dir, fname))
                       for fname in img_names]
            self.img_lst2.append(img_lst)

    def save(self, keyword):
        """
        Save the current images into new sub directories
        :param keyword: The string to append to the end of the original names for the
                        new sub directories that we are saving to
        """
        # Use the keyword to make the new names of the sub dirs
        # new_sub_dirs = ['%s.%s' % (sub_dir, keyword) for sub_dir in self.sub_dirs]
        new_sub_dirs = ['%s.%s' % (sub_dir, keyword)
                        for sub_dir in self.sub_dirs]
        new_sub_dirs_split = [os.path.split(
            new_sub_dir) for new_sub_dir in new_sub_dirs]
        new_sub_dirs = [os.path.join(new_sub_dir_split[0], keyword, new_sub_dir_split[1])
                        for new_sub_dir_split in new_sub_dirs_split]

        # Make new sub dir to hold all transformed sub_dirs (image folders)
        self._make_new_dir(os.path.split(new_sub_dirs[0])[0])

        # Loop through the sub dirs and loop through images to save images to the respective subdir
        for new_sub_dir, img_names, img_lst in zip(new_sub_dirs, self.img_names2, self.img_lst2):
            new_sub_dir_path = self._path_relative_to_parent(new_sub_dir)
            self._make_new_dir(new_sub_dir_path)

            for fname, img_arr in zip(img_names, img_lst):
                io.imsave(os.path.join(new_sub_dir_path, fname),
                          img_as_ubyte(img_arr))
        self.sub_dirs = new_sub_dirs

    def show(self, img_idx, sub_dir=None, sub_dir_idx=None):
        """
        View the nth image in the nth class
        :param sub_dir: The name of the category
        :param sub_dir_idx: The name of the category
        :param img_idx: The index of the category of images
        """
        if sub_dir_idx == None:
            sub_dir_ind = self.label_map[sub_dir]
            io.imshow(self.img_lst2[sub_dir_idx][img_idx])
            plt.show()
        elif sub_dir == None:
            print('\n displaying image... \n')
            io.imshow(self.img_lst2[sub_dir_idx][img_idx])
            plt.show()
        else:
            raise Exception('Specify only a sub_dir_idx or a sub_dir name')

    def transform(self, func, params, sub_dir=None, img_idx=None):
        """
        Takes a function and apply to every img_arr in self.img_arr.
        Have to option to transform one as  a test case
        :param sub_dir: The index for the image
        :param img_idx: The index of the category of images
        """
        # Apply to one test case
        if sub_dir is not None and img_idx is not None:
            sub_dir_ind = self.label_map[sub_dir]
            img_arr = self.img_lst2[sub_dir_ind][img_idx]
            img_arr = func(img_arr, **params).astype(float)
            io.imshow(img_arr)
            plt.show()
        # Apply the function and parameters to all the images
        else:
            new_img_lst2 = []
            for img_lst in self.img_lst2:
                new_img_lst2.append(
                    [func(img_arr, **params).astype(float) for img_arr in img_lst])
            self.img_lst2 = new_img_lst2

    def grayscale(self, sub_dir=None, img_idx=None):
        """
        Grayscale all the images in self.img_lst2
        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_idx: The index of the image within the chosen sub dir
        """
        self.transform(color.rgb2gray, {}, sub_dir=sub_dir, img_idx=img_idx)

    def canny(self, sub_dir=None, img_idx=None):
        """
        Apply the canny edge detection algorithm to all the images in self.img_lst2
        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_idx: The index of the image within the chosen sub dir
        """
        self.transform(feature.canny, {}, sub_dir=sub_dir, img_idx=img_idx)

    def tv_denoise(self, weight=2, multichannel=True, sub_dir=None, img_idx=None):
        """
        Apply to total variation denoise to all the images in self.img_lst2
        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_idx: The index of the image within the chosen sub dir
        """
        self.transform(restoration.denoise_tv_chambolle,
                       dict(weight=weight, multichannel=multichannel),
                       sub_dir=sub_dir, img_idx=img_idx)

    def resize(self, shape, save=False):
        """
        Resize all images in self.img_lst2 to a uniform shape
        :param shape: A tuple of 2 or 3 dimensions depending on if your images are grayscaled or not
        :param save: Boolean to save the images in new directories or not
        """
        self.transform(transform.resize, dict(output_shape=shape))
        if save:
            shape_str = '_'.join(map(str, shape))
            self.save(shape_str)

    def _vectorize_features(self):
        """
        Take a list of images and vectorize all the images. Returns a feature matrix where each
        row represents an image
        """
        row_tup = tuple(img_arr.ravel()[np.newaxis, :]
                        for img_lst in self.img_lst2 for img_arr in img_lst)
        self.test = row_tup
        self.features = np.r_[row_tup]

    def _vectorize_labels(self):
        """
        Convert file names to a list of y labels (in the example it would be either cat or dog, 1 or 0)
        """
        # Get the labels with the dimensions of the number of image files
        self.labels = np.concatenate([np.repeat(i, len(img_names))
                                      for i, img_names in enumerate(self.img_names2)])

    def vectorize(self):
        """
        Return (feature matrix, the response) if output is True, otherwise set as instance variable.
        Run at the end of all transformations
        """
        self._vectorize_features()
        self._vectorize_labels()
        return self.features, self.labels

    def prep_X_y(self):
        X = 0
        y = 0
        self.img_lst2
        for i in range(len(self.img_lst2)):
            #     if limit == None:
            #         images = load_image_folder(folder_list[i])
            #     else:
            #         images = load_image_folder(folder_list[i], limit)
            # cropped = crop_image_list(images, crop_size)
            # resized = self.resize(shape = resize_dim, save=False)
            # rot_crop = rotate_images_4x(cropped)
            # mirrored = mirror_images(rot_crop)
            # if False:
            if type(X) == int:
                X = np.array(self.img_lst2)
            else:
                X = np.vstack((X, np.array(self.img_lst2)))
            # if False:
            if type(y) == int:
                y = np.zeros(len(self.img_lst2))
            else:
                y_arr = np.zeros(len(self.img_lst2))
                y_arr.fill(i)
                y = np.append(y, y_arr)
        print('X shape: {} ----- y shape: {}'.format(X.shape, y.shape))

        return np.array(X), y


def load_and_featurize_data():
    # The data, split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # Convert class vectors to binary class matrices.
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return X_train, X_test, y_train, y_test


def define_model(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def compile_model(model, optimizer):
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def train_model(model, data_augmentation, batch_size, epochs,
                X_train, X_test, y_train, y_test):

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            workers=4,
                            steps_per_epoch=len(X_train) // batch_size,
                            use_multiprocessing=True)
    return model


def save_model(model, save_dir, model_name):
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def evaluate_model(model, X_test, y_test):
    # Score trained model.
    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':

    plt.rcParams.update({'font.size': 22})  # for Frank
    plt.style.use('ggplot')  # for not blue
    np.random.seed(42)  # for reproducibility

    # transformations = [rgb2gray, sobel, canny, denoise_tv_chambolle, denoise_bilateral]

    ip = ImagePipeline('../data/gray_32')
    print('Reading Images...\n')
    ip.read()
    # print('Resizing Images...')
    # ip.resize(shape=(16, 16, 3), save=False)  # 64=.37, 32=.36
    # print('Grayscaling Images...')
    # ip.grayscale()
    # ip.save('gray_16')

    print('Vectorizing...\n')
    features, target = ip.vectorize()
    print('splitting train/test \n')
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        random_state=42)
    
    # Random Forest
    rf_accuracy_lst = []
    rf_precision_lst = []
    rf_recall_lst = []
    rf_num_trees_list = list(np.arange(100,1000,20))
    rf_num_trees_list.extend(list(np.arange(1000,5000,200)))

    for num_trees in rf_num_trees_list:

        print('RF',num_trees,'Classifying...\n')
        rf = RandomForestClassifier(bootstrap=True,
                                    ccp_alpha=0.0,
                                    class_weight='balanced',  # default None
                                    criterion='gini',
                                    max_depth=None, #default None
                                    max_features='auto',  # None = +-8% of % long rt
                                    max_leaf_nodes=None,
                                    max_samples=None,
                                    min_impurity_decrease=0.0,
                                    min_impurity_split=None,
                                    min_samples_leaf=1,
                                    min_samples_split=2,
                                    min_weight_fraction_leaf=0.0,
                                    n_estimators=num_trees,  # two class=100=0.70, 1000=0.75,10k=0.74
                                    n_jobs=-2,  # use all CPUs but 1
                                    oob_score=True, # Use out-of-bag samples
                                    random_state=1,
                                    verbose=0,
                                    warm_start=False
                                    )
        print('RF Fitting...\n')
        rf.fit(X_train, y_train)
        print('Predicting...\n')
        rf_preds = rf.predict(X_test)
        print('Calculating Accuracy...\n')
        rf_accuracy_lst.append(accuracy_score(y_test, rf_preds))
        rf_precision_lst.append(precision_score(y_test, rf_preds))
        rf_recall_lst.append(recall_score(y_test, rf_preds))

    # rf_num_trees_list = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800]
    # rf_accuracy_lst = [0.6838709677419355, 0.6903225806451613, 0.6580645161290323, 0.6774193548387096, 0.6774193548387096, 0.6903225806451613, 0.6967741935483871, 0.7096774193548387, 0.7096774193548387, 0.6967741935483871, 0.7161290322580646, 0.7032258064516129, 0.7032258064516129, 0.7225806451612903, 0.7096774193548387, 0.7032258064516129, 0.7161290322580646, 0.7096774193548387, 0.7096774193548387, 0.7161290322580646, 0.7032258064516129, 0.6967741935483871, 0.6903225806451613, 0.6903225806451613, 0.6903225806451613, 0.7096774193548387, 0.7096774193548387, 0.6903225806451613, 0.6903225806451613, 0.6967741935483871, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7096774193548387, 0.7096774193548387, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6903225806451613, 0.6967741935483871, 0.6903225806451613, 0.7032258064516129, 0.7096774193548387, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.6967741935483871, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.6967741935483871, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.6967741935483871, 0.7096774193548387, 0.7096774193548387, 0.7032258064516129, 0.7096774193548387, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7096774193548387, 0.7032258064516129, 0.7096774193548387, 0.7032258064516129, 0.7032258064516129, 0.7096774193548387, 0.7161290322580646, 0.7161290322580646, 0.7225806451612903, 0.7161290322580646, 0.7225806451612903, 0.7290322580645161, 0.7096774193548387, 0.7032258064516129, 0.6967741935483871, 0.7032258064516129, 0.6967741935483871, 0.6967741935483871, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7032258064516129, 0.7096774193548387]
    # rf_precision_lst = [0.6990291262135923, 0.7019230769230769, 0.6796116504854369, 0.6923076923076923, 0.696078431372549, 0.7019230769230769, 0.7047619047619048, 0.7184466019417476, 0.7184466019417476, 0.7087378640776699, 0.7169811320754716, 0.7075471698113207, 0.7075471698113207, 0.7238095238095238, 0.7142857142857143, 0.7075471698113207, 0.7211538461538461, 0.7142857142857143, 0.7142857142857143, 0.7169811320754716, 0.7115384615384616, 0.7047619047619048, 0.7019230769230769, 0.6981132075471698, 0.6981132075471698, 0.7102803738317757, 0.7102803738317757, 0.6944444444444444, 0.7019230769230769, 0.7047619047619048, 0.7115384615384616, 0.7115384615384616, 0.7115384615384616, 0.7075471698113207, 0.7075471698113207, 0.7075471698113207, 0.7142857142857143, 0.7142857142857143, 0.7009345794392523, 0.6972477064220184, 0.7009345794392523, 0.7009345794392523, 0.7009345794392523, 0.6944444444444444, 0.7009345794392523, 0.6909090909090909, 0.6964285714285714, 0.7027027027027027, 0.6964285714285714, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6936936936936937, 0.7, 0.7, 0.7, 0.7, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.6972477064220184, 0.7, 0.7, 0.7, 0.7, 0.6972477064220184, 0.7027027027027027, 0.7027027027027027, 0.7, 0.7027027027027027, 0.7, 0.7, 0.7, 0.7027027027027027, 0.7, 0.7027027027027027, 0.7, 0.7, 0.7027027027027027, 0.7053571428571429, 0.7090909090909091, 0.719626168224299, 0.7129629629629629, 0.719626168224299, 0.7222222222222222, 0.7064220183486238, 0.7, 0.6936936936936937, 0.7, 0.6972477064220184, 0.6972477064220184, 0.7037037037037037, 0.7037037037037037, 0.7037037037037037, 0.7037037037037037, 0.7037037037037037, 0.7037037037037037, 0.7102803738317757]
    # rf_recall_lst = [0.8, 0.8111111111111111, 0.7777777777777778, 0.8, 0.7888888888888889, 0.8111111111111111, 0.8222222222222222, 0.8222222222222222, 0.8222222222222222, 0.8111111111111111, 0.8444444444444444, 0.8333333333333334, 0.8333333333333334, 0.8444444444444444, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8444444444444444, 0.8222222222222222, 0.8222222222222222, 0.8111111111111111, 0.8222222222222222, 0.8222222222222222, 0.8444444444444444, 0.8444444444444444, 0.8333333333333334, 0.8111111111111111, 0.8222222222222222, 0.8222222222222222, 0.8222222222222222, 0.8222222222222222, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8444444444444444, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8444444444444444, 0.8666666666666667, 0.8666666666666667, 0.8666666666666667, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8444444444444444, 0.8666666666666667, 0.8666666666666667, 0.8555555555555555, 0.8666666666666667, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8666666666666667, 0.8555555555555555, 0.8666666666666667, 0.8555555555555555, 0.8555555555555555, 0.8666666666666667, 0.8777777777777778, 0.8666666666666667, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8666666666666667, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8555555555555555, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444, 0.8444444444444444]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rf_num_trees_list, rf_accuracy_lst, label='Accuracy')
    ax.plot(rf_num_trees_list, rf_precision_lst, label='Precision')
    ax.plot(rf_num_trees_list, rf_recall_lst, label='Recall')
    ax.legend()
    plt.savefig('../img/rf-num-tree-10k-g32.png')


    # print('\n \n start trees list:', rf_num_trees_list)
    # print('\n \n start acc list:', rf_accuracy_lst)
    # print('\n \n start prec list:', rf_precision_lst)
    # print('\n \n start rec list:', rf_recall_lst)
    
        
        # rf_accuracy = accuracy_score(y_test, rf_preds)
        # rf_precision = precision_score(y_test, rf_preds)
        # rf_recall = recall_score(y_test, rf_preds)
        # print('\n \n RF Accuracy is: ', rf_accuracy, '\n')
        # print('\n Out-of-"Boot" Score is: ', rf.oob_score_, '\n')
        # print('\n RF Precision is: ', rf_precision, '\n')
        # print('\n RF Recall is: ', rf_recall, '\n \n')

    # print('\n Cross Validation Score 5 Folds:'
    # print('\n Avg Accuracy:',
    #         np.mean(cross_val_score(rf,X_train,y_train, scoring='accuracy')))
    # print('\n Avg Precision:',
    #         np.mean(cross_val_score(rf,X_train,y_train, scoring='precision')))
    # print('\n Avg Recall:',
    #         np.mean(cross_val_score(rf,X_train,y_train, scoring='recall')))
    

    # duration = 1  # seconds
    # freq = 440  # Hz
    # os.system('spd-say "your program has finished"')

    # ip.show(sub_dir_idx=0, img_idx=2)
    # ip.show(sub_dir_idx=0, img_idx=1)
    # ip.show(sub_dir_idx=2, img_idx=2)
    # ip.show(sub_dir_idx=2, img_idx=1)

    # X, y = ip.prep_X_y()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # convert class vectors to binary class matrices (don't change)
    # Y_train = np_utils.to_categorical(y_train, num_classes) # cool
    # Y_test = np_utils.to_categorical(y_test, num_classes)   # cool * 2

    # # new doc
    # # https://github.com/GalvanizeDataScience/lectures/blob/Denver/convolutional-neural-nets/frank-burkholder/keras_example_cifar10.py

    # X_train, X_test, y_train, y_test = load_and_featurize_data()
    # input_shape = X_train.shape[1:]

    # model = define_model(num_classes, input_shape)

    # model = compile_model(model, gd_optimizer)

    # model = train_model(model, data_augmentation, batch_size, epochs,
    #                     X_train, X_test, y_train, y_test)

    # save_model(model, save_dir, model_name)

    # evaluate_model(model, X_test, y_test)

    """
    Wood Capstone Doc
    """

    # model = Sequential()
    # model.add(Conv2D(nb_filters, (5, 5),
    #                     padding='valid',
    #                     input_shape=input_shape))
    # model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers
    # model.add(Conv2D(nb_filters, (3,3)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(nb_filters, (2,2)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    # model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    # model.add(Flatten()) # necessary to flatten before going into conventional dense layer (keep layer)
    # print('Model flattened out to ', model.output_shape)

    # # now start a typical neural network
    # model.add(Dense(128))
    # model.add(Activation('tanh'))
    # model.add(Dropout(0.15))
    # model.add(Dense(num_classes)) # 10 final nodes (one for each class) (keep layer)
    # model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-9
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])

    # # during fit process watch train and test error simultaneously
    # model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
    #           verbose=1, validation_data=(X_test, Y_test))

    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1]) # this is the one we care about

    """
    BUILDING THE (classless) IMAGE CLEANING PIPELINE
    """
    # img_folder = '../data/subset'

    # # read in an image of interest
    # path_string_list = os.listdir(img_folder)
    # path_string_list.sort()

    # # split path_string ('acura_TL_1984') on '_'
    # # path_list == list of lists (llist)
    # # eg. [[acura, TL, 1984],[acura, TL, 1985], [acura, GE, 1999]]
    # path_llist = [re.split("_", path_string) for path_string in path_string_list]

    # parent_dir = "../data/subset/"
    # img_name_llist = [os.listdir(parent_dir + path) for path in path_string_list]

    # img_path_name_dict = {}
    # for i, path in enumerate(path_string_list):
    #     if path not in img_path_name_dict:
    #         img_path_name_dict[path] = img_name_llist[i]

    # img_dict = {path: [] for path in path_string_list}
    # for path, fname_list in img_path_name_dict.items():
    #     for fname in fname_list:
    #         if path in img_dict:
    #             img_dict[path].append(io.imread(os.path.join(parent_dir, path, fname)))

    # io.imshow(img_dict['acura_cl_1997'][1])
    # plt.show()
