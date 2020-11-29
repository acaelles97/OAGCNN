from oagcnn.engine.predictor import OAGCNNPredictor
from oagcnn.engine.default_argparser import argument_parser
from oagcnn.models.oagcnn import OAGCNN
from oagcnn.config.defaults import cfg


def inference_from_config(cfg):
    device = "cuda" if cfg.PREDICTOR.USE_GPU else "cpu"
    model = OAGCNN(cfg, device)

    predictor = OAGCNNPredictor(cfg, device, model)
    predictor.run_predictor()


if __name__ == "__main__":
    args = argument_parser()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    inference_from_config(cfg)
