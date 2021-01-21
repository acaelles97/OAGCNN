from collections import namedtuple
import logging
import shutil
import os

class CheckPointer:
    def __init__(self, cfg, training_name, output_path):
        self._patience = cfg.PATIENCE
        self._patience_delta = cfg.PATIENCE_DELTA
        self._training_name = training_name
        self._patience_counter = 0
        self._save_always = cfg.SAVE_ALWAYS
        self._init_out_checkpoints_dir(output_path)
        self.logger = logging.getLogger("OAGCNN")

        self._epoch_recorder = namedtuple('EpochRecorder', ['epoch', 'loss'])
        self._best_epoch = None
        self.keep_training = True

    def _init_out_checkpoints_dir(self, output_path):
        self._checkpoints_save_dir = os.path.join(output_path, "checkpoints")
        if not os.path.exists(self._checkpoints_save_dir):
            os.makedirs(self._checkpoints_save_dir, exist_ok=True)

    def reset_patience_counter(self):
        self._patience_counter = 0

    def get_out_name_by_epoch(self, epoch):
        return "{}_epoch_{}.pth".format(self._training_name, epoch)

    def get_out_name_best(self, epoch):
        return self.get_out_name_by_epoch("{}_BEST".format(epoch))

    def update_val(self, epoch, new_val_loss):

        if self._best_epoch is None:
            self._best_epoch = self._epoch_recorder(epoch=epoch, loss=new_val_loss)
            out_name = self.get_out_name_best(epoch)
            return os.path.join(self._checkpoints_save_dir, out_name)

        elif new_val_loss < (self._best_epoch.loss - self._patience_delta):
            # Rename epoch with best on it's name
            last_out_path = os.path.join(self._checkpoints_save_dir, self.get_out_name_best(self._best_epoch.epoch))
            last_new_path = os.path.join(self._checkpoints_save_dir, self.get_out_name_by_epoch(self._best_epoch.epoch))
            shutil.move(last_out_path, last_new_path)

            # Save new best epoch
            self._best_epoch = self._epoch_recorder(epoch=epoch, loss=new_val_loss)
            out_path = os.path.join(self._checkpoints_save_dir, self.get_out_name_best(self._best_epoch.epoch))
            self.logger.info("NEW BEST EPOCH: {}".format(epoch))
            self._patience_counter = 0
            return out_path

        else:
            self._patience_counter += 1
            if self._patience_counter > self._patience:
                self.keep_training = False

            if self._save_always:
                return os.path.join(self._checkpoints_save_dir, self.get_out_name_by_epoch(epoch))

            else:
                return False
