from os.path import join, splitext
from os import listdir
from sys import argv
import re
import io

'''
Project Gutenberg books have junk at the beginning/throughout like Chapters/Notices, so I wanted to remove these. Another problem is that paragraphs are made of many line breaks. Paragraphs can be understood as samples, so we want a single sample per paragraph.
'''

clean_dir = '../data'

for f in listdir(clean_dir):
        # All Gutenberg books should be .txt files
        if not f.endswith('.txt'):
                continue

        full_path = join(clean_dir, f)

        # If you don't want to replace the book
        # out_path = join('books', splitext(f)[0] + '-cleaned.txt')
        out_path = full_path
        with io.open(full_path, 'r', encoding='utf-8') as fd:
                # Split up paragraphs, then combine all of each paragraph into a single line.
                # \n\n first line \n second line \n third line \n\n -> \n first line second line third line \n
                cleaned = "\n".join([' '.join([line.strip() for line in paragraph.split('\n')]) for paragraph in
                           fd.read().split('\n\n') if len(paragraph) > 20])

        with io.open(full_path, 'w+', encoding='utf-8') as out_fd:
                out_fd.write(cleaned)

        print('Cleaned ' + full_path)
