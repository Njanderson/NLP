from models.model import Model
import numpy as np
from math import log
from logging import Logger, getLogger, INFO
import sys
from utils.loader import read_bson, write_bson

logger = getLogger('Ngram')
Logger.setLevel(logger, INFO)

class Ngram(Model):

    def __init__(self, n, language, seed=1, smoothing=1, out=sys.stdout, counts={}):
        # The counts of all ngrams seem
        self.counts = counts
        # Save language
        self.language = language
        # The next possible chars and their probabilities
        self.chars = []
        self.probabilities = []
        # Discrete conditional distribution conditioned on history
        self.distrib = None
        # The n to use for ngrams
        self.n = n
        # The current history
        self.history = ''
        # The smoothing factor added to each character
        self.smoothing = smoothing
        # Use the same rand_state for every distribution
        np.random.seed(seed)
        # For testing, collect output
        self.out = out
        # Only leave the distribution null if we have to train our model
        if len(counts) > 0:
            self._create_dist()

    """
    Given a map of history to next character counts, construct a
    conditional probability distribution.
    """

    def _create_dist(self):
        # How long can our ngram be?
        ngram_length = self.n
        while len(self.history) < ngram_length:
            ngram_length -= 1

        # What is the longest ngram that we have seen?
        ngram_history = self.history[-ngram_length:]
        while ngram_history not in self.counts:
            ngram_history = ngram_history[1:]

        # These are the relevant counts
        # Make a deep copy to prevent running out of memory
        # by adding smoothing with every iteration
        ngram_counts = dict(self.counts[ngram_history])

        total_count = 0
        # Add smoothing such that we have a valid probability distribution
        for c in self.language:
            # c = c.decode('utf8')
            if c not in ngram_counts:
                ngram_counts[c] = 0
            ngram_counts[c] += self.smoothing
            total_count += ngram_counts[c]

        # Store chars and their probabilities
        self.chars = list(ngram_counts.keys())
        self.probabilities = [ngram_counts[c] / total_count for c in self.chars]

        # See most likely probability values
        # print(sorted(self.probabilities, reverse=True)[:10])

        logger.info('Created new distribution')

    """Train the model. Accepts data, a list of sentences to train on"""
    def train(self, data):
        logger.info('Training model...')
        # Used to print training progress
        # count = 0
        # log_freq = 100

        for sentence in data:
            # Print progress
            # if count % log_freq == 0:
            #     print('Trained on %d out of %d samples' % (count, len(data)))
            #     pass
            # count += 1

            # Add ^C character
            sample = sentence + chr(3)
            history = ''
            for c in sample:
                # If history is too long, truncate it
                if len(history) > self.n:
                    history = history[1:]

                curr = history
                while curr is not None:

                    # If this history has never been seen, add it
                    if curr not in self.counts:
                        self.counts[curr] = {}

                    # Update counts of this next character given this history
                    seen = self.counts[curr]
                    if c not in seen:
                        seen[c] = 0
                    seen[c] += 1

                    curr = curr[1:] if len(curr) > 0 else None

                history += c

        # Once all samples in the data have been parsed, write bson to disk
        write_bson(self.counts, type(self).__name__)

        logger.info('Trained model!')
        self._create_dist()

    """Generate a character the model"""
    def generate(self):
        # Sample from our distribution and index into our characters
        sampled = np.random.choice(self.chars, p=self.probabilities)
        print(sampled + ' generated (probability=%f). ' % (self.probabilities[self.chars.index(sampled)],), end='', file=self.out)
        self.observe(sampled)
        return sampled

    """Observe character "observed" and update history, returns the log probability of that character"""
    def observe(self, observed):
        print('Observed: ' + observed.replace('\n', '\\n'), file=self.out)
        if observed == chr(3):
            self.history = ''
        else:
            self.history += observed
        p = self.probabilities[self.chars.index(observed)]
        log_p = log(p) / log(2)
        self._create_dist()
        return log_p


    """Query the model for character "queried," prints and returns the log probability"""
    def query(self, queried, silent=False):
        p = self.probabilities[self.chars.index(queried)]
        log_p = log(p) / log(2)
        if not silent:
            print("%f" % (log_p, ), file=self.out)
        return log_p

    """Resets the history of the model"""
    def reset(self):
        self.history = ''
        self._create_dist()
