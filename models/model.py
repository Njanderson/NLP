class Model(object):

    """Train the model. Accepts data, a list of sentences to train on"""
    def train(self, language, data):
        raise NotImplementedError()

    """Generate a character the model"""
    def generate(self):
        raise NotImplementedError()

    """Observe character "observed" and update history, returns the log probability of that character"""
    def observe(self, observed):
        raise NotImplementedError()

    """Query the model for character "queried," prints and returns the log probability"""
    def query(self, queried):
        raise NotImplementedError()

    """Resets the history of the model"""
    def reset(self):
        raise NotImplementedError()
