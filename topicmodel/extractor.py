import os as Os
dirname = Os.path.dirname(__file__)
filename = Os.path.join(dirname, 'en09062011.news')

with open(filename) as raw_file:
    output_file = open(Os.path.join(dirname, 'en09062011_snippets.txt'), 'w')
    file_lines = raw_file.readlines() # read the file to the memory
    for i in range(len(file_lines)):
        if i % 8 == 1:
            output_file.write(file_lines[i])
            print(file_lines[i])
