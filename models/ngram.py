from models.model import Model
from scipy.stats import rv_discrete
import numpy as np
from math import log
from logging import Logger, getLogger, INFO

logger = getLogger('Ngram')
Logger.setLevel(logger, INFO)

class Ngram(Model):

    def __init__(self, n, seed=1, smoothing=1):
        # The counts of all ngrams seem
        self.counts = {}
        # All of the characters in the language
        self.language = []
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
        ngram_counts = self.counts[ngram_history]

        total_count = 0
        # Add smoothing such that we have a valid probability distribution
        for c in self.language:
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
    def train(self, language, data):
        logger.info('Training model...')
        self.language = language

        # Used to print training progress
        count = 0
        log_freq = 100

        for sentence in data:
            # Print progress
            if count % log_freq == 0:
                print('Trained on %d out of %d samples' % (count, len(data)))
            count += 1

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
        logger.info('Trained model!')
        self._create_dist()

    """Generate a character the model"""
    def generate(self):
        # Sample from our distribution and index into our characters
        sampled = np.random.choice(self.chars, p=self.probabilities)
        print(sampled + ' generated (probability=%f). ' % (self.probabilities[self.chars.index(sampled)],), end='')
        self.observe(sampled)

    """Observe character "observed" and update history"""
    def observe(self, observed):
        print('Observed: ' + observed)
        self.history += observed
        self._create_dist()

    """Query the model for character "queried" """
    def query(self, queried):
        p = self.probabilities[self.chars.index(queried)]
        print("%f" % (log(p) / log(2), ))

    """Resets the history of the model"""
    def reset(self):
        history = ''
        self._create_dist()
