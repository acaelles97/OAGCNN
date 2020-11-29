from yacs.config import CfgNode as CN
import os.path as osp

_C = CN()

_C.GENERAL_CONFIG = CN()
_C.GENERAL_CONFIG.TRAIN_NAME = "T00"
_C.GENERAL_CONFIG.PRINT_EVERY = 20
_C.GENERAL_CONFIG.USE_GPU = True

_C.PATH = CN()
# Root folder of project data?
_C.PATH.ROOT = osp.abspath('../../rvos/')
_C.PATH.MODELS = osp.abspath("/mnt/gpid07/users/adria.caelles/pre-trained_models")
_C.PATH.MODELS_RVOS_YOUTUBE = osp.join(_C.PATH.MODELS, "rvos_oneshot_youtubevos")
_C.PATH.MODELS_RVOS_DAVIS = osp.join(_C.PATH.MODELS, "rvos_oneshot_davis")

_C.PATH.OUTPUT_DIR = "/mnt/gpid07/users/adria.caelles/output_dir"

# Root folder of project datasets
_C.PATH.DATASETS = osp.abspath('/mnt/gpid07/users/adria.caelles/datasets')

_C.PATH.CONFIG_FILE = ""

# Color palette
_C.PATH.PALETTE = osp.abspath(osp.join(_C.PATH.ROOT, 'src/dataloader/palette.txt'))
_C.PATH.LMDB = osp.abspath('/mnt/gpid07/users/adria.caelles/datasets/lmdb_data')


_C.DATA = CN()
_C.DATA.BATCH_SIZE = 4
# Only 1 implemneted
_C.DATA.VAL_BATCH_SIZE = 1
_C.DATA.NUM_WORKERS = 1
_C.DATA.SHUFFLE = True
_C.DATA.DROP_LAST = True
_C.DATA.CLIP_LENGTH = 5
_C.DATA.MAX_NUM_OBJ = 5
_C.DATA.IMAGE_SIZE = (256, 448)
_C.DATA.AUGMENTATION = False

_C.AUG_TRANSFORMS = CN()
_C.AUG_TRANSFORMS.ROTATION = 10
_C.AUG_TRANSFORMS.TRANSLATION = 0.1
_C.AUG_TRANSFORMS.SHEAR = 0.1
_C.AUG_TRANSFORMS.ZOOM = 0.7

# DAVIS CONFIG
_C.Davis = CN()

# Name folder in DATABASES
_C.Davis.NAME = "DAVIS2017"

_C.Davis.BASE_PATH = osp.join(_C.PATH.DATASETS, _C.Davis.NAME)

# Path to input images
_C.Davis.IMAGES_FOLDER = "JPEGImages/480p/"
# Path to annotations
_C.Davis.ANNOTATIONS_FOLDER = "Annotations/480p/"
# PATH TO SEQUENCES_INFO
_C.Davis.DB_INFO = osp.abspath(osp.join(_C.PATH.ROOT, "src/dataloader/db_info.yaml"))

# YOUTUBE_VOS DATASET
_C.YouTubeVOS = CN()
_C.YouTubeVOS.NAME = "YouTubeVOS"
_C.YouTubeVOS.BASE_PATH = osp.join(_C.PATH.DATASETS, _C.YouTubeVOS.NAME)
# Path to input images
_C.YouTubeVOS.IMAGES_FOLDER = "JPEGImages"
_C.YouTubeVOS.ANNOTATIONS_FOLDER = "Annotations"
_C.YouTubeVOS.ADDITIONAL_PARTITIONS_FOLDER = "additional_partitions"

_C.TEST = CN()
_C.TEST.EVAL = CN()

# Path to property file, holding information on evaluation sequences.
# Metrics: J: region similarity, F: contour accuracy, T: temporal stability
_C.TEST.EVAL.METRICS = ['J', 'F']
# Statistics computed for each of the metrics listed above
_C.TEST.EVAL.STATISTICS = ['mean', 'recall', 'decay']
_C.TEST.EVAL_PERIOD = 1

_C.MODEL = CN()

# RVOSEncoder
_C.MODEL.FEATURE_EXTRACTOR = "RVOSFeatureExtractor"

_C.MODEL.RUNTIME_CONFIG = CN()
_C.MODEL.LOAD_CHECKPOINT = ""
_C.MODEL.RUNTIME_CONFIG.FreezeRVOSEncoder = 0
_C.MODEL.RUNTIME_CONFIG.UnfreezeRVOSEncoder = 2
_C.MODEL.RUNTIME_CONFIG.UsePreviousInferenceMask = 4

_C.RVOSFeatureExtractor = CN()

_C.RVOSFeatureExtractor.RVOS_ENCODER = CN()
_C.RVOSFeatureExtractor.RVOS_ENCODER.LOAD_PRETRAINED = True
# ResNet34, ResNet50, ResNet101, VGG16
_C.RVOSFeatureExtractor.RVOS_ENCODER.BACKBONE = "ResNet34"
_C.RVOSFeatureExtractor.RVOS_ENCODER.HIDDEN_SIZE = 128
_C.RVOSFeatureExtractor.RVOS_ENCODER.KERNEL_SIZE = 3


_C.RVOSFeatureExtractor.RVOS_ADAPTER = CN()
_C.RVOSFeatureExtractor.RVOS_ADAPTER.USED_FEATURES = ["x5", "x4", "x3", "x2"]
_C.RVOSFeatureExtractor.RVOS_ADAPTER.CHANNELS_FACTOR_REDUCTION = 2
# Spatial Scale WRT DATA.IMAGE_SIZE
_C.RVOSFeatureExtractor.RVOS_ADAPTER.SPATIAL_SCALE_FACTOR = 2

_C.RVOSFeatureExtractor.RVOS_HEAD = CN()
_C.RVOSFeatureExtractor.RVOS_HEAD.HIDDEN_SIZE = 256
_C.RVOSFeatureExtractor.RVOS_HEAD.DROPOUT = 0.5
_C.RVOSFeatureExtractor.RVOS_HEAD.OUT_CHANNELS = 8

_C.GCNN = CN()
_C.GCNN.MESSAGE_PASSING_STEPS = 3
_C.GCNN.MODULE = "GCNNSimple"
_C.GCNN.READ_OUT = "ReadOutSimple"
_C.GCNN.NUM_CLASSES = 1
_C.GCNN.LOSS_FUNCTION = "SoftIoULoss"
_C.GCNN.USE_PREVIOUS_INFERENCE_MASK = True
_C.GCNN.BACKPROPAGATE_PREDICTED_MASKS = True

_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.LR = 0.001

_C.SOLVER.EVALUATION_METRIC = "loss"
_C.SOLVER.CHECKPOINTER = CN()
_C.SOLVER.CHECKPOINTER.SAVE_ALWAYS = True
_C.SOLVER.CHECKPOINTER.PATIENCE = 5
_C.SOLVER.CHECKPOINTER.PATIENCE_DELTA = 0.1

_C.SOLVER.USE_SCHEDULER = True
_C.SOLVER.LR_SCHEDULER = CN()
_C.SOLVER.LR_SCHEDULER.NAME = "MultiStepLR"
_C.SOLVER.LR_SCHEDULER.STEPS = [6, 8]
_C.SOLVER.LR_SCHEDULER.STEP_SIZE = 0.5
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.NUM_EPOCHS = 10


_C.PREDICTOR = CN()
_C.PREDICTOR.USE_GPU = True
_C.PREDICTOR.OVERLAY_MASKS = True
_C.PREDICTOR.BASE_PATH_IMAGES = "/mnt/gpid07/users/adria.caelles/datasets/YouTubeVOS/valid/JPEGImages"
_C.PREDICTOR.BASE_PATH_ANNOTATIONS = "/mnt/gpid07/users/adria.caelles/datasets/YouTubeVOS/valid/Annotations"

_C.PREDICTOR.FOLDER_TO_INFER = ["6142dee608", "815744f806", "4e132f2efb"]


_C.DATASETS = CN()
# "YouTubeVOS" OR "Davis"
_C.DATASETS.TRAIN = "YouTubeVOS"
# "train" or "trainval"
_C.DATASETS.TRAIN_SPLIT = "train"
# Only for Davis
_C.DATASETS.TRAIN_YEAR = "2017"

_C.DATASETS.TEST = "YouTubeVOS"
# "test-dev" or "val"
_C.DATASETS.TEST_SPLIT = "val"

cfg = _C
