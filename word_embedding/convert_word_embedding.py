import my_utils
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Word Embedding')
    parser.add_argument('--glove', required=True, help='glove path')
    parser.add_argument('--save_path', required=True, help='save word2vec path')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    my_utils.convert_glove_to_word2vec(args.glove, args.save_path)
