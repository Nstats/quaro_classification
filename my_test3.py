from train import get_embedding_index


embedding_dir = './data/glove.840B.300d.txt'

embedding_index = get_embedding_index(embedding_dir)

word_list = ['formula', 'link']

for ele in word_list:
    print(ele in embedding_index)
