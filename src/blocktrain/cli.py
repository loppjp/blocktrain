import argparse

def cmd_init():
    print("INIT")
    raise NotImplementedError("Init not implemented")

def cmd_train():
    print("TRAIN")
    raise NotImplementedError("Train not implemented")

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

    #cmd_subparsers = parser.add_subparsers(title="init", dest="cmd_train")

    init_parser = cmd_subparsers.add_parser("init", help="i am the init function")

    train_parser = cmd_subparsers.add_parser("train", help="i am the train function")

    parser = parser.parse_args()

    CMD_MAP[parser.cmd]()

if __name__ == "__main__":
    main()