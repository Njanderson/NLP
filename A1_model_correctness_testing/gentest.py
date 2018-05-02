class Gentest(object):
    def __init__(self):
        self.VALID_BMP = [(0, 0x0870), (0x089F+1, 0x1C90), (0x1CBF+1, 0x2FE0), (0x2FEF+1, 65536)]

    def is_valid_bmp(self, index):
        return any(index >= a and index < b for a, b in self.VALID_BMP)

    def check(self, file_name):
        fh = open(file_name, 'r')
        lines = fh.readlines()
        for line in lines:
            ret = self.is_valid_bmp(ord(line[0]))
            print(ret)
        fh.close()

if __name__ == '__main__':
    file_name = "gen_output.txt"
    obj = Gentest()
    obj.check(file_name)
