import io
from os import listdir, path
from utils.const import LANG_MAX_VALUE

def get_language():
    lang = []
    for i in range(LANG_MAX_VALUE):
        if 0xD800 <= i <= 0xDFFF:
            continue
        lang.append(chr(i))
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