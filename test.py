from oagcnn.config.defaults import cfg
from oagcnn.models.oagcnn import OAGCNN
from oagcnn.engine import Trainer, argument_parser
from oagcnn.engine.evaluation import Evaluator
import torch

def test(model_path):
    device = "cuda" if cfg.GENERAL_CONFIG.USE_GPU else "cpu"
    model = OAGCNN(cfg, device)
    model_weights = torch.load(model_path)
    model.custom_load_state_dict(model_weights)

    evaluator = Evaluator(cfg, model, device)
    val_loss = evaluator.validate(0)
    print(val_loss)


if __name__ == "__main__":
    model_path = "/mnt/gpid07/users/adria.caelles/output_dir/T02_GCNN-TRAINING/checkpoints/T02_GCNN-TRAINING_epoch_5.pth"
    config_file = "/home/usuaris/imatge/adria.caelles/workdir/OAGCNNforMVOS/T04_inference.yaml"
    cfg.merge_from_file(config_file)
    cfg.freeze()
    test(model_path)