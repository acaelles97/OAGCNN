# This script generates a data structure in the form of key-value storage. This is made in the huge amount of 
# calls to the function os.listdir inside base_youtube.py
from oagcnn.data.dataset_creator import YouTubeVOSFactory, DavisFactory
from oagcnn.data.lmdb_generator import LMDBGenerator
from oagcnn.config.defaults import cfg
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='LMDB_GENERATOR')
    parser.add_argument('-dataset', help="youtube or davis. Check paths on overall config file")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = get_args()
    lmdb_dir = cfg["PATH"]["LMDB"]

    if args.dataset == 'youtube':
        base_dir = cfg["YouTubeVOS"]["BASE_PATH"]

        frame_lmdb_generator_sequences = LMDBGenerator(ext='.jpg', out_dir=lmdb_dir, dataset="YouTubeVOS")

        train_sequences = YouTubeVOSFactory.get_images_dir(base_dir, "train", cfg)
        test_sequences = YouTubeVOSFactory.get_images_dir(base_dir, "test", cfg)

        frame_lmdb_generator_sequences.generate_lmdb_file("train_sequences", train_sequences)
        frame_lmdb_generator_sequences.generate_lmdb_file("test_sequences", test_sequences)

        train_annot = YouTubeVOSFactory.get_annotations_dir(base_dir, "train", cfg)
        test_annot = YouTubeVOSFactory.get_annotations_dir(base_dir, "test", cfg)

        frame_lmdb_generator_annotations = LMDBGenerator(ext='.png', out_dir=lmdb_dir,  dataset="YouTubeVOS")
        frame_lmdb_generator_annotations.generate_lmdb_file("train_annotations", train_annot)
        frame_lmdb_generator_annotations.generate_lmdb_file("test_annotations", test_annot)
        
    else:
        base_dir = cfg["Davis"]["BASE_PATH"]

        frame_lmdb_generator_sequences = LMDBGenerator(ext='.jpg', out_dir=lmdb_dir, dataset="Davis")

        train_sequences = DavisFactory.get_images_dir(base_dir, "train", cfg)
        test_sequences = DavisFactory.get_images_dir(base_dir, "test", cfg)

        frame_lmdb_generator_sequences.generate_lmdb_file("train_sequences", train_sequences)
        frame_lmdb_generator_sequences.generate_lmdb_file("test_sequences", test_sequences)

        train_annot = DavisFactory.get_annotations_dir(base_dir, "train", cfg)
        test_annot = DavisFactory.get_annotations_dir(base_dir, "test", cfg)

        frame_lmdb_generator_annotations = LMDBGenerator(ext='.png', out_dir=lmdb_dir, dataset="Davis")
        frame_lmdb_generator_annotations.generate_lmdb_file("train_annotations", train_annot)
        frame_lmdb_generator_annotations.generate_lmdb_file("test_annotations", test_annot)
