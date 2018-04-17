from argparse import ArgumentParser
from models import ngram, cnn
from utils import loader
from sys import stdin
from logging import Logger, getLogger, INFO

logger = getLogger('main')
Logger.setLevel(logger, INFO)

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                    help="Model to use", required=True)
parser.add_argument("-d", "--data", dest="data",
                    help="The directory corresponding to training data")
parser.add_argument("-s", "--seed", dest="seed",
                    help="The random seed to be used")

args = parser.parse_args()


# Default to data directory
data = args.data if args.data is not None else 'data'
seed = args.seed

models = {
    'n': ngram.Ngram(5, int(seed), smoothing=0.01),
    'c': cnn.Cnn()
}
model = models[args.model]

if __name__ == '__main__':
    model.train(loader.get_language(), loader.load_all(data))
    while True:
        cmd = stdin.read(1)
        if cmd == 'o':
            model.observe(stdin.read(1))
        elif cmd == 'g':
            model.generate()
        elif cmd == 'q':
            model.query(stdin.read(1))
        elif cmd == 'x':
            exit(0)
        else:
            # Malformed input
            logger.warn('Malformed command')
