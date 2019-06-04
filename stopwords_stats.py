import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

train_dir = './data/train.csv'

df = pd.DataFrame(pd.read_csv(train_dir, engine='python'))

question_text = [word_tokenize(line) for line in df['question_text'].fillna('').values]
target = df['target']

# example = {'word': {'total': 100, 'insincere': 2, 'ratio': 0.02}}
stoplist = stopwords.words('english')

word_dict = {}
for stopword in stoplist:
    for i in range(len(target)):
        if stopword in question_text[i]:
            if stopword not in word_dict:
                word_dict.update({stopword: {'total': 1, 'insincere': 0, 'ratio': 0.}})
            else:
                word_dict[stopword]['total'] += 1
            if target[i] == 1:
                word_dict[stopword]['insincere'] += 1

for word in word_dict:
    if word_dict[word]['insincere'] == 0:
        word_dict[word]['ratio'] = 1e10
    else:
        word_dict[word]['ratio'] = (word_dict[word]['total'] - word_dict[word]['insincere'])/(word_dict[word]['insincere'])

sorted_word_dict = sorted(word_dict.items(), key=lambda item: item[1]['ratio'], reverse=True)
sorted_word_dict2 = sorted(word_dict.items(), key=lambda item: item[1]['insincere'], reverse=True)
sorted_word_dict3 = sorted(word_dict.items(), key=lambda item: item[1]['total'], reverse=True)

with open('./data/word_stats/stopwords_stats.txt', 'w') as f:
    for ele in sorted_word_dict:
        f.write(str(ele)+'\n')
