import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize, toimage
from PIL import Image


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



PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]



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

    def save_result(self, height, width, out_masks, frame_name, sequence_name, num_instances_clip, instances_id):
        num_masks = out_masks.shape[0]

        for t in range(num_masks):
            mask_pred = out_masks[t, ...]
            mask_pred = np.reshape(mask_pred, (height, width))
            mask_pred = mask_pred * 255

            instance_id = instances_id[t] + 1

            sub_folder = "instance_result/{}_instances".format(num_instances_clip)
            out_folder = os.path.join(self.output_path, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)

            output_path = os.path.join(out_folder, sequence_name[0])
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)

            filename = '{}_instance_{:02d}.png'.format(frame_name[0], instance_id)
            full_path = os.path.join(output_path, filename)

            toimage(mask_pred, cmin=0, cmax=255).save(full_path)

    def save_result_overlay(self, image, out_masks, frame_name, sequence_name, num_instances_clip):
        # image = np.rollaxis(image, 2, 0)
        height = image.shape[0]
        width = image.shape[1]

        plt.figure()
        plt.axis('off')
        plt.figure()
        plt.axis('off')
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

        sub_folder = "mask_overlay/{}_instances".format(num_instances_clip)
        out_folder = os.path.join(self.output_path, sub_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        output_path = os.path.join(out_folder, sequence_name[0])
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        fig_name = os.path.join(output_path, "{}.png".format(frame_name[0]))
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close()

    def _save_submission_image(self, image, folder_name, sequence_name, frame_name):
        res_im = Image.fromarray(image, mode="P")
        res_im.putpalette(PALETTE)

        out_folder = os.path.join(self.output_path, folder_name)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        output_path = os.path.join(out_folder, sequence_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        frame_path = os.path.join(output_path, "{}.png".format(frame_name))
        res_im.save(frame_path)

    def results_for_submission(self, height, width, out_masks, frame_name, sequence_name, instances_id):
        # LOAD GT
        # 1 compute area from out_masks
        num_objs = out_masks.shape[0]
        # First we print bigger areas
        sorted_idx = np.argsort([-np.sum(out_masks[i, ...]) for i in range(num_objs)])
        pred_mask_resized = np.zeros((height, width), dtype=np.uint8)

        for idx in sorted_idx:
            instance_id = instances_id[idx] + 1
            pred_mask = out_masks[idx, ...]

            pred_mask_resized_aux = imresize(pred_mask, (height, width), interp='nearest')
            pred_mask_resized[pred_mask_resized_aux == 255] = instance_id

        self._save_submission_image(pred_mask_resized, "submission_results", sequence_name[0], frame_name[0])

    def empty_mask_for_submission(self, height, width, frames_names, sequence_name):
        for frame_name in frames_names:
            pred_mask = np.zeros((height, width), dtype=np.uint8)
            self._save_submission_image(pred_mask, "submission_results", sequence_name[0], frame_name)


