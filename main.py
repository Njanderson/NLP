from argparse import ArgumentParser
from models import ngram, cnn
from utils import loader
from sys import stdin
from logging import Logger, getLogger, INFO

# TODO: We probably need this
from codecs import getreader
# Thanks to: https://stackoverflow.com/questions/2737966/how-to-change-the-stdin-encoding-on-python
# UTF8Reader = getreader('utf8')
# stdin = UTF8Reader(stdin)

logger = getLogger('main')
Logger.setLevel(logger, INFO)

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                    help="Model to use", required=True)
parser.add_argument("-d", "--data", dest="data",
                    help="The directory corresponding to training data")
parser.add_argument("-t", "--t", dest="train",
                    help="Whether to train the model or not. Changes how data is interpretted from being a cached dictionary to a directory of files")
parser.add_argument("-s", "--seed", dest="seed",
                    help="The random seed to be used")

# Pass in stream such that we can support stdin or file interfaces,
# returns True if execution can continue
def do_cmd(model, stream):
    cmds = stream.read()
    i = 0
    while i < len(cmds):
        cmd = cmds[i]
        i += 1
        if cmd == 'o':
            model.observe(cmds[i])
            i += 1
        elif cmd == 'g':
            model.generate()
        elif cmd == 'q':
            model.query(cmds[i])
            i += 1
        elif cmd == 'x':
            return
        else:
            # Malformed input
            logger.warning('Malformed command ' + cmd)

if __name__ == '__main__':
    args = parser.parse_args()

    # Default to data directory
    data = args.data if args.data is not None else 'data'
    seed = args.seed
    train = args.train is not None

    # TODO: If we make more models, we have to make sure they don't all run, like they do here. Maybe lazy evaluation.
    # Probably not this but something like this: https://github.com/janrain/lazydict would work if we wanted it.
    models = {
        'n': ngram.Ngram(5, loader.get_language(), int(seed), smoothing=0.01, counts=None if train else loader.read_bson(data)),
        'c': cnn.Cnn()
    }
    model = models[args.model]

    if train:
        model.train(loader.load_all(data))
    else:
        print('Skipping training...')

    do_cmd(model, stdin)

