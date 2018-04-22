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

# Pass in stream such that we can support stdin or file interfaces,
# returns True if execution can continue
def do_cmd(model, stream):
    try:
        # Read 1 character at a time
        cmd = stream.read(1)
        if cmd == 'o':
            model.observe(stream.read(1))
        elif cmd == 'g':
            model.generate()
        elif cmd == 'q':
            model.query(stream.read(1))
        elif cmd == 'x':
            return False
        else:
            # Malformed input
            logger.warning('Malformed command ' + cmd)
    except:
        return False
    return True

if __name__ == '__main__':
    args = parser.parse_args()

    # Default to data directory
    data = args.data if args.data is not None else 'data'
    seed = args.seed

    models = {
        'n': ngram.Ngram(5, int(seed), smoothing=0.01),
        'c': cnn.Cnn()
    }
    model = models[args.model]

    model.train(loader.get_language(), loader.load_all(data))

    while do_cmd(model, stdin):
        pass

