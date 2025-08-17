import argparse

from blocktrain.train import main as train_main
from blocktrain.utilities import get_config_path

def cmd_init(**kwargs):
    print("INIT")
    raise NotImplementedError("Init not implemented")

def cmd_train(args):
    train_main(
        **vars(args)
    )

CMD_MAP = {
    "init": cmd_init,
    "train": cmd_train
}

def main():
    help="BlockTrain CLI is a component based ML training framework and library"
    parser = argparse.ArgumentParser(help)

    # inspo https://stackoverflow.com/questions/10448200/how-to-parse-multiple-nested-sub-commands-using-python-argparse/#answer-19476216
    parser.add_argument('cmd')

    cmd_subparsers = parser.add_subparsers(title="blocktrain", dest="cmd_init")

    init_parser = cmd_subparsers.add_parser("init", help="init function")

    train_parser = cmd_subparsers.add_parser("train", help="train function")

    train_parser.add_argument(
        "-e",
        "--experiment-path",
        dest="experiment_path",
        type=str,
        default=get_config_path()/"experiment.yaml",
        help="path to the experiment config file"
    )

    parser_map = {
        "train": train_parser,
        "init": init_parser
    }

    cmd_parser_args = parser.parse_args()

    args = parser_map[cmd_parser_args.cmd].parse_known_args()[0]

    CMD_MAP[cmd_parser_args.cmd](args)

if __name__ == "__main__":
    main()