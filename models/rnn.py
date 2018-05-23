from models.model import Model
import numpy as np

class Rnn(Model):

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        # randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        
    """Train the model. Accepts data, a list of sentences to train on"""
    def train(self, language, data):
        raise NotImplementedError()

    """Generate a character the model"""
    def generate(self):
        raise NotImplementedError()

    """Observe character "observed" and update history"""
    def observe(self, observed):
        raise NotImplementedError()

    """Query the model for character "queried" """
    def query(self, queried):
        raise NotImplementedError()

    """Resets the history of the model"""
    def reset(self):
        raise NotImplementedError()
