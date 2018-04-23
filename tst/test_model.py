import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import unittest
import utils
import io
from main import do_cmd
from models.ngram import Ngram
from math import *
import tempfile
from re import *

# How long do we expect sentences could be?
MAX_EXPECTED_LENGTH = 500
TOLERANCE = 1e-5
SEED = 1
TEST_DATA_DIR = os.path.join('resources', 'data')
TEST_CMD_DIR = os.path.join('resources', 'cmd')
NUM_THREADS = 8

class TestModels(unittest.TestCase):

    def output_helper(self, expected, found):
        # Make sure our output has the same length as the reference output
        self.assertEqual(expected.count('\n'), found.count('\n'))
        expected_split = expected.split('\n')
        found_split = found.split('\n')
        # Check to make sure lines that report a number report in our model as well
        for i in range(0, len(expected_split)):
            if match('-?[0-9]', expected_split[i]) is not None:
                self.assertTrue(match('-?[0-9]', found_split[i]))

    """Evaluates on assignment 1 test input"""

    def basic_helper(self, model, out):
        lang = utils.loader.get_language()
        with io.open(os.path.join(TEST_CMD_DIR, 'basic_input'), 'r', encoding='utf-8') as fd:
            # Perform a command from the file stream
            while do_cmd(model, fd):
                # After performing a command, check the probability
                self.assertAlmostEqual(sum(model.probabilities), 1.0)
                self.assertTrue(set(lang).issubset(model.chars))
        with io.open(os.path.join(TEST_CMD_DIR, 'basic_output'), 'r') as fd:
            expected = fd.read()
        out.seek(0)
        found = out.read()
        self.output_helper(expected, found)

    """Evaluates whether a stop character is eventually produced"""

    def termination_helper(self, model):
        # Generate a single character
        # Note: we loop here such that if we generate a ^C as our first character,
        # we still test this functionality
        sentence = ''
        try:
            while len(model.history) == 0:
                sentence += model.generate()
            while len(model.history) > 0:
                sentence += model.generate()
                if len(model.history) > MAX_EXPECTED_LENGTH:
                    self.fail('Failed to generate sentence ending before %d characters' % (MAX_EXPECTED_LENGTH,))
        finally:
            print('Sentence generated: ' + sentence)

    def test_ngram_basic(self):
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as tmp:
            model = Ngram(5, seed=int(SEED), smoothing=0.01, out=tmp)
            model.train(utils.loader.get_language(), utils.loader.load_all(TEST_DATA_DIR))
            self.basic_helper(model, tmp)

    def test_ngram_termination(self):
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as tmp:
            model = Ngram(5, seed=int(SEED), smoothing=1e-10, out=tmp)
            model.train(utils.loader.get_language(), utils.loader.load_all(TEST_DATA_DIR))
            self.termination_helper(model)

if __name__ == '__main__':
    unittest.main()
