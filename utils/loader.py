import io
from os import listdir, path
from utils.const import LANG_MAX_VALUE
import bson
from os.path import join

def read_bson(filename):
    out = join('data', '%s' % (filename,))
    with open(out, "rb") as fd:
        return bson.loads(fd.read())

def write_bson(to_write, filename):
    out = join('data', '%s.bson' % (filename,))
    with open(out, "wb+") as fd:
        fd.write(bson.dumps(to_write))

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
        if not f.endswith('.txt'):
            continue
        full_path = path.join(data_root, f)
        samples += get_samples(full_path)
    return samples

