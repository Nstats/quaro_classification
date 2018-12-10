from collections import Counter
from train import preprocessing
from train import get_embedding_index

train_dir = './data/train.csv'
embedding_dir = './data/glove.840B.300d.txt'
words_not_in_file_dir = './data/word_stats/words_not_in_glove_after_preprocessed.txt'

rectify_dict = {'Blockchain': 'blockchain', 'b.SC': 'b.sc', 'Cryptocurrency': 'cryptocurrency',
                "Qur'an": 'Quoran', 'etc…': 'etc', 'Unacademy': 'unacademic', 'Quorans': 'Quoran',

                'colour': 'color', 'centre': 'center', 'favourite': 'favorite',
                'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization',
                'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
                'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much',
                'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best',
                'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018',
                'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'
                }

df, _ = preprocessing(train_dir, rectify_dict, train_or_dev=False)
question_text = df['question_text']

embedding_index = get_embedding_index(embedding_dir)

kinds_total = []
kinds_not_have_vec = []

total_num = 0
words_not_have_vec = 0
words_not_in_file = []

for line in question_text:
    for word in line:
        total_num += 1
        if word not in kinds_total:
            kinds_total.append(word)
        if word not in embedding_index:
            if word not in kinds_not_have_vec:
                kinds_not_have_vec.append(word)
            words_not_have_vec += 1
            words_not_in_file.append(word)

words_not_in_file_stats = Counter(words_not_in_file).most_common()
with open(words_not_in_file_dir, 'w') as f:
    f.write('Training set has total {0} words and {1}% of them do NOT have corresponding word vector.'
            .format(total_num, (words_not_have_vec/total_num)*100)+'\n')
    f.write('Training set has total {0} KIND of words and {1}% of them do Not have corresponding word vector.'
            .format(len(kinds_total), (len(kinds_not_have_vec)/len(kinds_total))*100)+'\n')
    for ele in words_not_in_file_stats:
        T_or_F = ele[0].lower() in embedding_index
        f.write(str(ele)+'  .lower() in embedding? '+str(T_or_F)+'\n')
