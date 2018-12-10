embedding_dir = './data/glove.840B.300d.txt'
words_in_file_dir = './data/words_in_glove.txt'

with open(words_in_file_dir, 'w') as file_out:
    with open(embedding_dir, 'r') as file_in:
        for line in file_in:
            file_out.write(line.split(' ')[0]+'\n')