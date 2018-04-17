import io
from os import listdir, path
from utils.const import LANG_SIZE

def get_language():
    lang = []
    for i in range(0,  LANG_SIZE):
        # TODO Are there ranges that should be skipped?
        lang += [chr(i)]
    return lang

def get_samples(path):
    with io.open(path, 'r', encoding='utf-8') as fd:
        return fd.read().split('\n')

def load_all(data_root):
    samples = []
    for f in listdir(data_root):
        full_path = path.join(data_root, f)
        samples += get_samples(full_path)
    return samples