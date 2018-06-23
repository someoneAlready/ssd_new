from __future__ import print_function, absolute_import
import os
import numpy as np
from .imdb import Imdb
import xml.etree.ElementTree as ET
from evaluate.eval_voc import voc_eval
import cv2
import PIL


class Psdb(Imdb):
    """
    Implementation of Imdb for Pascal VOC datasets

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    year : str
        year of dataset, can be 2007, 2010, 2012...
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, devkit_path, shuffle=False, is_train=False, class_names=None,
            names='psdb.names'):
        super(Psdb, self).__init__('Psdb_' + image_set)
        self.image_set = image_set
        self.devkit_path = devkit_path
        self.data_path = devkit_path
        self.extension = '.jpg'
        self.is_train = is_train

        if class_names is not None:
            self.classes = class_names.strip().split(',')
        else:
            self.classes = self._load_class_names(names,
                os.path.join(os.path.dirname(__file__), 'names'))

        self.config = {'use_difficult': True,
                       'comp_id': 'comp4',}

        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()


    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _filter_image_with_no_gt(self):
        """
        filter images that have no ground-truth labels.
        use case: when you wish to work only on a subset of pascal classes, you have 2 options:
            1. use only the sub-dataset that contains the subset of classes
            2. use all images, and images with no ground-truth will count as true-negative images
        :return:
        self object with filtered information
        """

        # filter images that do not have any of the specified classes
        self.labels = [f[np.logical_and(f[:, 0] >= 0, f[:, 0] <= self.num_classes-1), :] for f in self.labels]
        # find indices of images with ground-truth labels
        gt_indices = [idx for idx, f in enumerate(self.labels) if not f.size == 0]

        self.labels = [self.labels[idx] for idx in gt_indices]
        self.image_set_index = [self.image_set_index[idx] for idx in gt_indices]
        old_num_images = self.num_images
        self.num_images = len(self.labels)

        print ('filtering images with no gt-labels. can abort filtering using *true_negative* flag')
        print ('... remaining {0}/{1} images.  '.format(self.num_images, old_num_images))

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.data_path, self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, 'image', name)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.data_path, 'annotation', 
                index[:-3] + 'txt')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        # load ground-truth from xml annotations


        for i, idx in enumerate(self.image_set_index):

            im_name = self.image_path_from_index(i)
            width, height = self._get_imsize(im_name)

            label = []
            label_file = self._label_path_from_index(idx)

            lines = np.loadtxt(label_file, delimiter='\n', dtype=str)[1:]
            for line in lines:
                line = line.strip().split()
                line = map(float, line)

                for j, cls_name in enumerate(self.classes):

                    if cls_name not in self.classes:
                        cls_id = len(self.classes)
                    else:
                        cls_id = self.classes.index(cls_name)

                    box = line[1 + j * 5: (j + 1) * 5]
                    if np.array(box).prod() < 0:
                        continue
                    x1, y1, w, h = box
                    x2 = x1 + w
                    y2 = y1 + h
                    xmin = float(x1) / width
                    ymin = float(y1) / height
                    xmax = float(x2) / width
                    ymax = float(y2) / height
                    difficult = 0

                    label.append([cls_id, xmin, ymin, xmax, ymax, difficult])
            temp.append(np.array(label))

        return temp




    def _get_imsize(self, im_name):
        """
        get image size info
        Returns:
        ----------
        tuple of (height, width)
        """
        return PIL.Image.open(im_name).size
#        img = cv2.imread(im_name)
#        return (img.shape[0], img.shape[1])
