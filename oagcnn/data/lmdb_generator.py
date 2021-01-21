import os
import lmdb


# LMDB is created for a sequence KEY->Sequence name / VALUE->all the files inside the folder
class LMDBGenerator:
    TYPES = {"train_sequences", "test_sequences", "train_annotations", "test_annotations"}

    def __init__(self, ext='.jpg', out_dir="", dataset=""):
        self.ext = ext
        self.out_dir = out_dir
        self.dataset = dataset

    @staticmethod
    def get_lmdb_read_path(dataset, lmdb_type):
        if lmdb_type not in LMDBGenerator.TYPES:
            RuntimeError("LMDB TYPES ALLOWED ARE: {} AND RECIEVED: {}".format(LMDBGenerator.TYPES, lmdb_type))
        return "lmdb_{}_{}".format(dataset, lmdb_type)

    def generate_lmdb_file(self, lmdb_type, frames_dir):
        out_filename = self.get_lmdb_read_path(self.dataset, lmdb_type)
        env = lmdb.open(os.path.join(self.out_dir, out_filename))
        root_in_dirs = os.listdir(frames_dir)

        for d in root_in_dirs:
            folder_dir = os.path.join(frames_dir, d)

            _files_basename = sorted([f for f in os.listdir(folder_dir) if f.endswith(self.ext)])
            files_str_vec = '|'.join(_files_basename)

            print("Generating lmdb for: " + folder_dir)
            with env.begin(write=True) as txn:
                txn.put(d.encode('ascii'), files_str_vec.encode())