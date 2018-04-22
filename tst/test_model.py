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
MAX_EXPECTED_LENGTH = 100
TOLERANCE = 1e-5
SEED = 1
TEST_DATA_DIR = os.path.join('resources', 'data')
TEST_CMD_DIR = os.path.join('resources', 'cmd')

class TestModels(unittest.TestCase):

    """Verifies probability sums to 1"""
    def probability_helper(self, model):
        p = 0.0
        i = 0
        log_freq = 10000
        lang = utils.loader.get_language()
        for c in lang:
            if i % log_freq == 0:
                print('Computing probability: %d of %d characters completed...' % (i, len(lang)))
            i += 1
            log_p = model.query(c)
            p += exp(log_p * log(2))
        return p

    def output_helper(self, expected, found):
        # Make sure our output has the same length as the reference output
        self.assertEquals(expected.count('\n'), found.count('\n'))
        expected_split = expected.split('\n')
        found_split = found.split('\n')
        # Check to make sure lines that report a number report in our model as well
        for i in range(0, len(expected.split)):
            if match('-?[0-9]', expected[i]) is not None:
                self.assertTrue(match('-?[0-9]', found[i]))

    """Evaluates on assignment 1 test input"""
    def basic_helper(self, model, out):
        with io.open(os.path.join(TEST_CMD_DIR, 'basic_input'), 'r') as fd:
            # Perform a command from the file stream
            do_cmd(model, fd)
            # After performing a command, check the probability
            self.assertTrue(isclose(1.0, self.probability_helper(model), abs_tol=TOLERANCE))
        with io.open(os.path.join(TEST_CMD_DIR, 'basic_output'), 'r') as fd:
            expected = fd.read()
        found = out.read()
        self.output_helper(expected, found)

    """Evaluates whether a stop character is eventually produced"""
    def termination_helper(self, model):
        # Generate a single character
        # Note: we loop here such that if we generate a ^C as our first character,
        # we still test this functionality
        while len(model.history) == 0:
            model.generate()
        while len(model.history) > 0:
            model.generate()
            if len(model.history) > MAX_EXPECTED_LENGTH:
                self.fail('Failed to generate sentence ending before %d characters' % (MAX_EXPECTED_LENGTH,))

    def test_ngram_basic(self):
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as tmp:
            model = Ngram(5, seed=int(SEED), smoothing=0.01, out=tmp)
            model.train(utils.loader.get_language(), utils.loader.load_all(TEST_DATA_DIR))
            self.basic_helper(model, tmp)

    def test_ngram_termination(self):
        model = Ngram(5, seed=int(SEED), smoothing=0.01)
        model.train(utils.loader.get_language(), utils.loader.load_all(TEST_DATA_DIR))
        self.termination_helper(model)

if __name__ == '__main__':
    unittest.main()