import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize, toimage


def sequence_palette():
    # RGB to int conversion

    palette = {(0, 0, 0): 0,
               (0, 255, 0): 1,
               (255, 0, 0): 2,
               (0, 0, 255): 3,
               (255, 0, 255): 4,
               (0, 255, 255): 5,
               (255, 128, 0): 6,
               (102, 0, 102): 7,
               (51, 153, 255): 8,
               (153, 153, 255): 9,
               (153, 153, 0): 10,
               (178, 102, 255): 11,
               (204, 0, 204): 12,
               (0, 102, 0): 13,
               (102, 0, 0): 14,
               (51, 0, 0): 15,
               (0, 64, 0): 16,
               (128, 64, 0): 17,
               (0, 192, 0): 18,
               (128, 192, 0): 19,
               (0, 64, 128): 20,
               (224, 224, 192): 21}

    return palette


class SaveResults:
    def __init__(self, save_path, model_name):

        self._init_output_paths(save_path, model_name)
        # Colors for overlay
        self.colors = self._init_colors()

    def _init_output_paths(self, out_dir, model_name):
        out_folder_name = "{}_inference".format(model_name)
        output_path = os.path.join(out_dir, out_folder_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path

    @staticmethod
    def _init_colors():
        colors = []
        palette = sequence_palette()
        inv_palette = {}
        for k, v in palette.items():
            inv_palette[v] = k
        num_colors = len(inv_palette.keys())
        for id_color in range(num_colors):
            if id_color == 0 or id_color == 21:
                continue
            c = inv_palette[id_color]
            colors.append(c)

        return colors

    def save_result(self, image, out_masks, frame_name, sequence_name):
        height = image.shape[-2]
        width = image.shape[-1]

        num_masks = out_masks.shape[0]
        for t in range(num_masks):
            mask_pred = out_masks[t, ...]
            mask_pred = np.reshape(mask_pred, (height, width))
            indxs_instance = np.where(mask_pred > 0.5)

            mask2assess = np.zeros((height, width))
            mask2assess[indxs_instance] = 255
            mask_save_path = os.path.join(self.output_path, frame_name + '_instance_{}.png'.format(t))

            toimage(mask2assess, cmin=0, cmax=255).save(mask_save_path)

    def save_result_overlay(self, image, out_masks, frame_name, sequence_name):
        # image = np.rollaxis(image, 2, 0)
        height = image.shape[0]
        width = image.shape[1]

        plt.figure()
        plt.axis('off')
        plt.figure()
        plt.axis('off')
        image = image / 255
        plt.imshow(image)
        num_masks = out_masks.shape[0]

        for t in range(num_masks):
            mask_pred = out_masks[t, ...]
            mask_pred = np.reshape(mask_pred, (height, width))
            ax = plt.gca()
            tmp_img = np.ones((mask_pred.shape[0], mask_pred.shape[1], 3))
            color_mask = np.array(self.colors[t]) / 255.0
            for i in range(3):
                tmp_img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((tmp_img, mask_pred * 0.7)))

        out_folder = os.path.join(self.output_path, sequence_name)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        fig_name = os.path.join(out_folder, "{}.png".format(frame_name))
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close()