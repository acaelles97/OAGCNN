import errno
import hashlib
import os
import os.path as osp
import sys
import tarfile
import h5py
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import time
from scipy.misc import imresize
import random
from .transforms.transforms import Affine
import glob
import json
from .transforms.transforms import CustomComposeTransform
import lmdb
from .clip_dataset import SequenceDataset
from args import get_parser

from abc import abstractmethod


# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()

if args.dataset == 'youtube':
    from misc.config_youtubeVOS import cfg as cfg_youtube
else:
    from misc.config import cfg


class MVOSDataset(data.Dataset):
    def __init__(self,
                 images_dir,
                 annotations_dir,
                 lmdb_dir):

        # To configure via gin
        # self.max_seq_len = config.gt_maxseqlen
        # self._length_clip = config.length_clip
        # self.augment = augment
        # self.dataset = config.dataset
        # self.dataset_config = config.dataset
        # self.use_prev_mask = use_prev_mask
        # self.base_dir = input_path
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.clips = None
        self.load_metadata()
        self.lmdb_dir = lmdb_dir

    def load_metadata(self):
        metadata = self.read_metadata()
        self._metadata = metadata

    @abstractmethod
    def read_metadata(self):
        pass


    # Returns tuple full path of specific clip identifier
    def get_raw_sample_clip(self, identifier):
        images_path = os.path.join(self.images_dir, identifier)
        annotations_path = os.path.join(self.annotations_dir, identifier)

        return images_path, annotations_path

    def get_lmdb_env(self):
        # First check lmdb existence.
        lmdb_env_images = None
        lmdb_env_annotations = None
        lmdb_env_seq_dir = osp.join(self.lmdb_dir, 'lmdb_seq')
        lmdb_env_annot_dir = osp.join(self.lmdb_dir, 'lmdb_annot')
        if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
            lmdb_env_images = lmdb.open(lmdb_env_seq_dir)
            lmdb_env_annotations = lmdb.open(lmdb_env_annot_dir)

        return lmdb_env_images, lmdb_env_annotations

    def create_clips(self):
        lmdb_env_images, lmdb_env_annotations = self.get_lmdb_env()

        clips = []
        for sequence in self._metadata.keys():
            frames_with_objs_in_sequence =  self._metadata[sequence]["Frames_with_instances"]
            obj_info_per_frame =  self._metadata[sequence]["Instance_categories"]
            images_path, annotations_path = self.get_raw_sample_clip(sequence)
            clip_transform = CustomComposeTransform(self.dataset_config)
            clip_dataset = SequenceDataset(images_path, annotations_path, frames_with_objs_in_sequence, obj_info_per_frame, clip_transform, lmdb_env_images, lmdb_env_annotations)
            clips.append(clip_dataset)

        self.clips = clips




# Necessito dir on mirar sequences / dir on mirar annotations / metadata amb key sequence
# Info basifca de treure de youtube -> Com llegir el metadata i la informació que trec de ell
class YouTubeVOS(MVOSDataset):

    def __init__(self, annotations_dir, images_dir, lmdb_dir, split, name, config_yaml):
        self.split = split
        self.name = name
        self.config_yaml = config_yaml
        super().__init__(images_dir, annotations_dir, lmdb_dir)



    # For each video -> Annotations, Sequences and metadata as dict of key sequence  / value info
    # We needs to check starting frame of each sequence this is shit
    def read_metadata(self):
        # Estic a Davis, per tant puc pillar la config de Davis directament
        metadata_path = self.config_yaml[self.name][self.split]
        metadata = json.load(metadata_path)

        # Save frames and num_instances each frame has
        # Sequences name

        metadata_info = {}
        # Check if any video has no annotations on the first frame
        for sequence in metadata["videos"].keys():
            sequence_info = {}
            for obj_id  in  metadata["videos"][sequence]["objects"].keys():
                obj_cat = metadata["videos"][sequence]["objects"][obj_id]["category"]
                frames = metadata["videos"][sequence]["objects"][obj_id]["category"]
                for frame in frames:
                    if frame not in sequence_info.keys():
                        sequence_info[frame] = []
                    sequence_info[frame].append(obj_cat)
            metadata_info[sequence]["Frames_with_instances"] = sequence_info.keys()
            metadata_info[sequence]["Instance_categories"] = sequence_info.values()

        return metadata_info

class Davis(MVOSDataset):

    def __init__(self, annotations_dir, images_dir, split, lmdb_dir, name, config_yaml, base_dir):
        self.split = split
        self.name = name
        self.config_yaml = config_yaml
        self.base_dir = base_dir
        super().__init__(images_dir, annotations_dir, lmdb_dir)



    # For each video -> Annotations, Sequences and metadata as dict of key sequence  / value info
    # We needs to check starting frame of each sequence this is shit
    def read_metadata(self):
        metadata_info = {}
        with open(self.config_yaml[self.name][self.split], 'r') as f:
            # Key sequence: Value obj:id value cat name
            categories = json.load(f)

        subset_path = os.path.join(self.base_dir, "ImageSets", self.name , self.split + '.txt')
        with open(subset_path, 'r') as f:
            sequences_in_split = f.readlines()
        # Sequences on the desired split
        sequences = [x.strip() for x in sequences_in_split]

        for sequence in sequences:
            sequence_info = {}
            dir_annotations = os.path.join(self.annotations_dir, sequence)
            list_valid_images = []
            for annotation in os.listdir(dir_annotations):
                annotation_image = Image.open(os.path.join(dir_annotations, annotation))
                num_instances = np.unique(annotation_image)

                if num_instances > 1:
                    list_valid_images.append(annotation)


        # Save frames and num_instances each frame has
        # Sequences name

        metadata_info = {}
        # Check if any video has no annotations on the first frame
        for sequence in metadata["videos"].keys():
            sequence_info = {}
            for obj_id  in  metadata["videos"][sequence]["objects"].keys():
                obj_cat = metadata["videos"][sequence]["objects"][obj_id]["category"]
                frames = metadata["videos"][sequence]["objects"][obj_id]["category"]
                for frame in frames:
                    if frame not in sequence_info.keys():
                        sequence_info[frame] = []
                    sequence_info[frame].append(obj_cat)
            metadata_info[sequence]["Frames_with_instances"] = sequence_info.keys()
            metadata_info[sequence]["Instance_categories"] = sequence_info.values()

        return metadata_info



# Fare Davis amb la info de Davis especifica per llegir el meta de Davis



# Useful to delimiter what we want/need to read from the metadata of each type of Dataset (i.e. Davis or YouTube-VOS)
class MetadataDatasetInfo:
    def __init__(self):
