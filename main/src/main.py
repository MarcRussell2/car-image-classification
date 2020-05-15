print('\n \n Importing...\n')
import os
import re
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})  # for Frank
plt.style.use('ggplot')  # for not blue
np.random.seed(42)  # for reproducibility

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


def plot_accuracies(df):
    fig = plt.figure(figsize=(8,4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)


    ax.plot(df.num_trees, 
            (100*df.accuracy_gray_16), 
            label = '16x16',
            color = '#8feaff'
            )

    ax.plot(df.num_trees, 
            (100*df.accuracy_gray_32), 
            label = '32x32',
            color = '#04add4'
            )

    ax.plot(df.num_trees, 
            (100*df.accuracy_gray_64), 
            label = '64x64',
            color = '#00748f'
            )

    hline_height = 100 * df[['accuracy_gray_16', 
                                    'accuracy_gray_32', 
                                    'accuracy_gray_64']].max().max()
    plt.axhline(y=hline_height, linewidth=1,linestyle='--', color='k', alpha=0.3)
    plt.axvline(x=860, linewidth=3,linestyle='--', color='k', alpha=0.15)

    plt.title('Accuracy vs Number of Trees Across Three Image Scales')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy                \nPercentage                 ', rotation='horizontal')
    plt.legend(title=' Image\nResolution')
    plt.tight_layout()
    plt.savefig('../img/rf-num-tree-10k-acc.png')

def plot_precisions(df):
    fig = plt.figure(figsize=(8,4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)


    ax.plot(df.num_trees, 
            (100*df.precision_gray_16), 
            label = '16x16',
            color = '#ffb65c'
            )

    ax.plot(df.num_trees, 
            (100*df.precision_gray_32), 
            label = '32x32',
            color = '#f08400'
            )

    ax.plot(df.num_trees, 
            (100*df.precision_gray_64), 
            label = '64x64',
            color = '#824800'
            )

    hline_height = 100 * df[['precision_gray_16', 
                                    'precision_gray_32', 
                                    'precision_gray_64']].max().max()
    plt.axhline(y=hline_height, linewidth=1,linestyle='--', color='k', alpha=0.3)
    plt.axvline(x=850, linewidth=2.5,linestyle='--', color='k', alpha=0.15)

    plt.title('Precision vs Number of Trees Across Three Image Scales')
    plt.xlabel('Number of Trees')
    plt.ylabel('Precision                \nPercentage                 ', rotation='horizontal')
    plt.legend(title=' Image\nResolution')
    plt.tight_layout()
    plt.savefig('../img/rf-num-tree-10k-prec.png')


def plot_recalls(df):
    fig = plt.figure(figsize=(8,4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)


    ax.plot(df.num_trees, 
            (100*df.recall_gray_16), 
            label = '16x16',
            color = '#fabaff'
            )

    ax.plot(df.num_trees, 
            (100*df.recall_gray_32), 
            label = '32x32',
            color = '#bb00c9'
            )

    ax.plot(df.num_trees, 
            (100*df.recall_gray_64), 
            label = '64x64',
            color = '#670070'
            )

    hline_height = 100 * df[['recall_gray_16', 
                                    'recall_gray_32', 
                                    'recall_gray_64']].max().max()
    plt.axhline(y=hline_height, linewidth=1,linestyle='--', color='k', alpha=0.3)

    plt.axvline(x=888, linewidth=4.5,linestyle='--', color='k', alpha=0.15)

    plt.title('Recall vs Number of Trees Across Three Image Scales')
    plt.xlabel('Number of Trees')
    plt.ylabel('Recall                  \nPercentage                 ', rotation='horizontal')
    plt.legend(title=' Image\nResolution')
    plt.tight_layout()
    plt.savefig('../img/rf-num-tree-10k-rec.png')

def Create_Model(df, transformed_sub_dirs, shape=(64, 64, 3),resize=False):
    for t_sub_dir in transformed_sub_dirs:
        ip = ImagePipeline(os.path.join('../data/',t_sub_dir))
        print('Reading Images...\n')
        ip.read()
        if resize == True:
            print('Resizing Images...')
            ip.resize(shape=shape, save=False)  # 64=.37, 32=.36
            print('Grayscaling Images...')
            ip.grayscale()
            ip.save('gray_64')

        print('Vectorizing...\n')
        features, target = ip.vectorize()
        print('splitting train/test \n')
        X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                            test_size=0.2,
                                                            random_state=42)
        rf_accuracy_lst = []
        rf_precision_lst = []
        rf_recall_lst = []
        rf_num_trees_list = list(np.arange(100,1000,20))
        rf_num_trees_list.extend(list(np.arange(1000,5000,200)))

        for num_trees in rf_num_trees_list:
            print('RF',num_trees,'Classifying...', t_sub_dir,'\n')
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

        '''
        making the df to store the plotted values
        '''
        col_name = 'accuracy_'
        col_name += t_sub_dir
        df[col_name] = rf_accuracy_lst

        col_name = 'precision_'  
        col_name += t_sub_dir
        df[col_name] =  rf_precision_lst

        col_name = 'recall_'
        col_name += t_sub_dir
        df[col_name] = rf_recall_lst

    df['num_trees'] = rf_num_trees_list
    df.to_csv('../data/numerical_data/scores_gray_x.csv')
    df.to_pickle('../data/numerical_data/scores_gray_x.pkl')


if __name__ == '__main__':
    print('\nReading...\n')
    Create_Model(df=df_eval_trans, transformed_sub_dirs = ['gray_16', 'gray_32', 'gray_64'], s=(64,64,3))

    df_eval_trans = pd.read_pickle('../data/numerical_data/scores_gray_x.pkl')

    print('plotting...\n\n')
    plot_accuracies(df_eval_trans)
    plot_precisions(df_eval_trans)
    plot_recalls(df_eval_trans)




 