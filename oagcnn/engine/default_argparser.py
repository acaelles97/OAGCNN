import argparse

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    # parser.add_argument(
    #     "--resume",
    #     action="store_true",
    #     help="Whether to attempt to resume from the checkpoint directory. "
    #     "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    # )
    # parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    # parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    # parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    # parser.add_argument(
    #     "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    # )
    #
    # # PyTorch still may leave orphan processes in multi-gpu training.
    # # Therefore we use a deterministic way to obtain port,
    # # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    # parser.add_argument(
    #     "--dist-url",
    #     default="tcp://127.0.0.1:{}".format(port),
    #     help="initialization URL for pytorch distributed backend. See "
    #     "https://pytorch.org/docs/stable/distributed.html for details.",
    # )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args
