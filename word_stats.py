import pandas as pd
from nltk.tokenize import word_tokenize

train_dir = './data/train.csv'

df = pd.DataFrame(pd.read_csv(train_dir, engine='python'))
question_text = [word_tokenize(line) for line in df['question_text'].values]
target = df['target']

example = {'word': {'total': 100, 'insincere': 2, 'ratio': 0.02}}

word_dict = {}
for i in range(len(target)):
    for word in question_text[i]:
        if word not in word_dict:
            word_dict.update({word: {'total': 1, 'insincere': 0, 'ratio': 0.}})
        else:
            word_dict[word]['total'] += 1
        if target[i] == 1:
            word_dict[word]['insincere'] += 1

for word in word_dict:
    word_dict[word]['ratio'] = word_dict[word]['insincere'] / word_dict[word]['total']

sorted_word_dict = sorted(word_dict.items(), key=lambda item: item[1]['ratio'], reverse=True)
sorted_word_dict2 = sorted(word_dict.items(), key=lambda item: item[1]['insincere'], reverse=True)
sorted_word_dict3 = sorted(word_dict.items(), key=lambda item: item[1]['total'], reverse=True)

with open('./data/sorted_by_ratio.txt', 'w') as f:
    for ele in sorted_word_dict:
        f.write(str(ele)+'\n')

with open('./data/sorted_by_insincere.txt', 'w') as f:
    for ele in sorted_word_dict2:
        f.write(str(ele)+'\n')

with open('./data/sorted_by_total.txt', 'w') as f:
    for ele in sorted_word_dict3:
        f.write(str(ele)+'\n')
