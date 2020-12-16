from oagcnn.config.defaults import cfg
from oagcnn.models.oagcnn import OAGCNN
from oagcnn.engine import Trainer, argument_parser


def train(cfg, args):

    device = "cuda" if cfg.GENERAL_CONFIG.USE_GPU else "cpu"

    model = OAGCNN(cfg, device)

    trainer = Trainer(cfg, model, device)

    if args.resume:
        trainer.resume_training(args.checkpoint)

    else:
        trainer.train()



if __name__ == "__main__":
    args = argument_parser()
    cfg.merge_from_file(args.config_file)
    cfg.PATH.CONFIG_FILE = args.config_file
    cfg.freeze()
    train(cfg, args)

