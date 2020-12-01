from oagcnn.data.transforms.custom_transforms import CustomRandomAffine, Affine
from torchvision.transforms import ToTensor, Resize, Normalize
from torchvision.transforms import functional as F
import random
from PIL import Image
from oagcnn.config.defaults import cfg
import torch
import numpy as np


# Wraps the call functionality of transforms to apply simultaneously to both image and annotation
class CustomComposeTransform:
    def __init__(self, augment):
        self.resize_image = Resize(cfg.DATA.IMAGE_SIZE, interpolation=Image.BILINEAR)
        self.resize_annotation = Resize(cfg.DATA.IMAGE_SIZE, interpolation=Image.NEAREST)
        self.to_tensor = ToTensor()
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.augment = augment
        self.custom_affine = None
        if augment:
            self.create_tf_matrix = CustomRandomAffine(rotation_range=cfg.AUG_TRANSFORMS.ROTATION,
                                           translation_range=cfg.AUG_TRANSFORMS.TRANSLATION,
                                           shear_range=cfg.AUG_TRANSFORMS.SHEAR,
                                           zoom_range=(cfg.AUG_TRANSFORMS.ZOOM, max(cfg.AUG_TRANSFORMS.ZOOM * 2, 1.0)),
                                           interp='nearest',
                                           lazy=True)

    def initial_transform(self, image, annotation):
        if random.random() < 0.5:
            image = F.hflip(image)
            annotation = F.hflip(annotation)

        image = self.resize_image(image)
        annotation = self.resize_annotation(annotation)

        image = self.to_tensor(image)
        if self.augment:
            annotation = self.to_tensor(annotation)

            # We need to compute tf_matrix yet
            if self.custom_affine is None:
                tf_matrix = self.create_tf_matrix(image)
                self.custom_affine = Affine(tf_matrix, interp='nearest')

            image, annotation = self.custom_affine(image, annotation)

        image = self.normalize(image)

        return image, annotation

    def final_transform(self, annotation):
        annotation = self.to_tensor(annotation)

        return annotation

    def initial_annot_test_transform(self, annotation):
        return self.resize_annotation(annotation)

    def final_annot_test_transform(self, annotation, valid_masks):
        return self.to_tensor(annotation), self.to_tensor(valid_masks)

    def test_image_transform(self, image):
        return self.normalize(self.to_tensor(self.resize_image(image)))

    # def final_image_test_transform(self, image):
    #     return self.normalize(self.to_tensor(image))
    #
    # def initial_image_test_transform(self, image):
    #     return self.resize_image(image)
