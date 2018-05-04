import math
import numpy
import sys


class Ngram(object):
    # TODO: Implement this class
    def __init__(self):
        self.probs = {}
        self.probs_sum = 0
        self.n = 2

    def param_loader(self):
        pass

    def get_prob(self, history):
        return 0.0001

    def get_probs(self):
        pass

    def get_probs_sum(self):
        pass

    def get_n(self):
        return self.n


class LmMain:
    def __init__(self):
        self.ngramModel = Ngram()
        self.o_cmd = 'o'
        self.q_cmd = 'q'
        self.g_cmd = 'g'
        self.x_cmd = 'x'
        self.o_indicator = False
        self.q_indicator = False
        self.end_token = u"\u0003"
        self.history = u"\u0003"
        self.n = self.ngramModel.get_n()

    def model_setup(self):
        self.ngramModel.param_loader()

    def operate(self):
        instream = sys.stdin.read()

        for c in instream:
            if self.o_indicator:
                if c != self.end_token:
                    if len(self.history) < self.n:
                        self.history += c
                    else:
                        self.history = self.history[1:] + c
                    sys.stdout.write("// added a character to the history!" + '\n')
                else:
                    self.history = u"\u0003"
                    sys.stdout.write("// cleared the history!" + '\n')

                self.o_indicator = False

            elif self.q_indicator:
                ngram_prob = self.ngramModel.get_prob(c)
                sys.stdout.write(repr(math.log(ngram_prob, 2))+'\n')
                self.q_indicator = False

            else:
                if c == self.o_cmd:
                    self.o_indicator = True
                elif c == self.q_cmd:
                    self.q_indicator = True
                elif c == self.g_cmd:
                    # TODO: Random generation
                    random_char = 1
                    sys.stdout.write(chr(random_char) + " // generated a character!" + '\n')
                    if len(self.history) < self.n:
                         self.history += random_char
                    else:
                        self.history = self.history[1:] + random_char
                elif c == self.x_cmd:
                    exit(1)


if __name__ == '__main__':
    objLM = LmMain()
    objLM.model_setup()
    seed = int(sys.argv[1])
    numpy.random.seed(seed)
    objLM.operate()
