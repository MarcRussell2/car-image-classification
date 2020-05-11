import os
import re
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')
plt.grid(None)
np.random.seed(42)  # for reproducibility

from skimage import color, transform, restoration, io, feature

# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.preprocessing.image import ImageDataGenerator






# !
# IMPLIMENT THAT FUCKIN CLASS YO!
# !

class ImagePipeline(object):

    def __init__(self, parent_dir):
        """
        Manages reading, transforming and saving images
        :param parent_dir: Name of the parent directory containing all the sub directories
        """
        # Define the parent directory
        self.parent_dir = parent_dir

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
        return os.path.join(cur_path, self.parent_dir, some_dir)

    def _make_new_dir(self, new_dir):
        """
        Make a new sub directory with fully defined path relative to the parent directory
        :param new_dir: The name of a new sub dir
        """
        # Make a new directory for the new transformed images
        new_dir = self._path_relative_to_parent(new_dir)
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
        filtered_sub_dir = list(filter(self._accepted_dir_name, self.raw_sub_dir_names))
        self.sub_dirs = map(self._path_relative_to_parent, filtered_sub_dir)
        

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

                img_lst = [io.imread(os.path.join(sub_dir, fname)) for fname in img_names]
                self.img_lst2.append(img_lst)

    def save(self, keyword):
        """
        Save the current images into new sub directories
        :param keyword: The string to append to the end of the original names for the
                        new sub directories that we are saving to
        """
        # Use the keyword to make the new names of the sub dirs
        new_sub_dirs = ['%s.%s' % (sub_dir, keyword) for sub_dir in self.sub_dirs]

        # Loop through the sub dirs and loop through images to save images to the respective subdir
        for new_sub_dir, img_names, img_lst in zip(new_sub_dirs, self.img_names2, self.img_lst2):
            new_sub_dir_path = self._path_relative_to_parent(new_sub_dir)
            self._make_new_dir(new_sub_dir_path)

            for fname, img_arr in zip(img_names, img_lst):
                io.imsave(os.path.join(new_sub_dir_path, fname), img_arr)

        self.sub_dirs = new_sub_dirs

    def show(self, sub_dir, img_ind):
        """
        View the nth image in the nth class
        :param sub_dir: The name of the category
        :param img_ind: The index of the category of images
        """
        sub_dir_ind = self.label_map[sub_dir]
        io.imshow(self.img_lst2[sub_dir_ind][img_ind])
        plt.show()

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


if __name__=='__main__':
    ip = ImagePipeline('../data/subset')
    ip.read()
    ip.show('acura_cl_1997',0)


    """
    BUILDING THE IMAGE CLEANING PIPELINE
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