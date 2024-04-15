import argparse

from src import train


def main():
    parser = argparse.ArgumentParser(description="SMILES-X")
    parser.add_argument("config", help="config fileを読み込む")
    parser.add_argument("--gpus")
    args = parser.parse_args()
    config_filepath = args.config
    n_gpus = args.gpus
    train.run(config_filepath, n_gpus)


if __name__ == "__main__":
    main()
