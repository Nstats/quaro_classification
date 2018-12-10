import pandas as pd
import re
import time

start_time = time.time()
train_dir = './data/train.csv'

train_df = pd.DataFrame(pd.read_csv(train_dir, engine='python'))


def remove_link_and_slash_split(sen):
    regex_link = r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    regex_slash_3 = r'[a-zA-Z]{3,}/[a-zA-Z]{3,}/[a-zA-Z]{3,}'
    regex_slash_2 = r'[a-zA-Z]{3,}/[a-zA-Z]{3,}'
    sen, number = re.subn(regex_link, ' link ', sen)
    result = re.findall(regex_slash_3, sen, re.S)
    for word in result:
        new_word = word.replace('/', ' / ')
        sen = sen.replace(word, new_word)
    result = re.findall(regex_slash_2, sen, re.S)
    for word in result:
        new_word = word.replace('/', ' / ')
        sen = sen.replace(word, new_word)
    return sen


question_text = train_df['question_text'].values
size = train_df.shape[0]
for i in range(size):
    question_text[i] = remove_link_and_slash_split(question_text[i])
train_df['question_text'] = question_text

print('time used=', time.time()-start_time)
