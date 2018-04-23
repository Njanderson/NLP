import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from argparse import ArgumentParser
from utils import loader
from models import ngram, cnn
from tempfile import TemporaryFile

TEST_DATA_DIR = os.path.join('resources', 'test')
TRAIN_DATA_DIR = os.path.join('resources', 'train')

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                    help="Model to use", required=True)


if __name__ == '__main__':
    # Capture output into file to prevent spam
    with TemporaryFile(mode='w+', encoding='utf-8') as tmp:
        args = parser.parse_args()
        models = {
            'n': ngram.Ngram(5, smoothing=0.01, out=tmp),
            'c': cnn.Cnn()
        }
        model = models[args.model]

        # Train on all training data
        model.train(loader.get_language(), loader.load_all(TRAIN_DATA_DIR))

        # Read in all test data
        test_samples = []
        for f in os.listdir(TRAIN_DATA_DIR):
            full_path = os.path.join(TRAIN_DATA_DIR, f)
            test_samples += loader.get_samples(full_path)

        perplexity_total = 0.0

        for sample in test_samples:
            # Add ending char
            sample += chr(3)
            log_p_sum = 0.0
            for c in sample:
                # Observe returns the log_2(probability) that char c was observed
                log_p_sum += model.observe(c)
            model.reset()
            perplexity = 2**(-log_p_sum/len(sample))
            print("Perplexity: %f" % (perplexity,))
            perplexity_total += perplexity

        print("Average perplexity: %f" % (perplexity_total/len(test_samples),))