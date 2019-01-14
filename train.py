import tensorflow as tf
import tensorflow.contrib as tc
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import time
import re

start_time = time.time()

batch_size = 512
n_iteration = 20000
word_vec_len = 300
rnn_hidden_size = 150
rnn_layer_num = 2
MLP_hidden_layer = 150
dev_size = 20000
# lr = 5e-5
lr = 0.001

tf.flags.DEFINE_integer('q_max_len', 50, 'question max valid length')
tf.flags.DEFINE_float('dp_keep_prob', 0.2, 'df_keep_prob')
tf.flags.DEFINE_integer('ratio_1_0', 2, 'label_1*ratio_1_0 : label_0 in training set')
FLAGS = tf.flags.FLAGS

train_dir = './data/train_example.csv'
test_dir = './data/test.csv'
embedding_dir = './data/glove.840B.300d_example.txt'
# size of glove: 2196017
tensorboard_log_dir = \
    './tensorboard/preprocessed/ql'+str(FLAGS.q_max_len)+'_bs512_h150_lr5e-5_dp_prob'+str(FLAGS.dp_keep_prob)\
    + '_ratio1_0_'+str(FLAGS.ratio_1_0)


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

'''
assign_to_variable_list = \
    ['cryptocurrencies', 'Brexit', 'Redmi', '^2', 'x^2', 'OnePlus',
'UCEED', 'GDPR', 'demonetisation', '\\infty', 'Coinbase', 'Adityanath', 'Machedo', 'BNBR', 'Boruto', 'DCEU',
'ethereum', 'IIEST', 'alt-right', 'e^', 'SJWs', '..', 'A+', 'LNMIIT', 'Upwork', '10+', '.net', 'Zerodha',
'x+1', 'anti-Trump', 'x^', 'Kavalireddi', 'bhakts', 'Doklam', 'Vajiram', '.NET', 'NICMAR', '100+', 'chsl', 'MUOET',
'AlShamsi', '=0', '^3', 'HackerRank']
assign_to_variable_dict = \
    {'random_vec': tf.get_variable('random_vec', [word_vec_len], tf.float32, tf.contrib.layers.xavier_initializer())}
for ele in assign_to_variable_list:
    assign_to_variable_dict.update(
        {ele: tf.get_variable(ele.replace('.', 'zz').replace('+', '').replace('/', '').replace('\\', '')
                              .replace('=', 'equal').replace('^', ''),
                              [word_vec_len], tf.float32, tf.contrib.layers.xavier_initializer())})
'''


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


def remove_formula(sen):
    for i in range(100):
        judge1 = '[math]' in sen and '[/math]' in sen
        judge2 = '[math]' in sen and '[\math]'in sen
        judge3 = '[math]' in sen and '[math]'in sen.replace('[math]', '', 1)
        if judge1 or judge2:
            index1 = sen.find('[math]')
            index2 = max(sen.find('[\math]'), sen.find('[/math]'))+7
            (index1, index2) = (min(index1, index2), max(index1, index2))
            sen = sen.replace(sen[index1: index2], ' formula ')
        elif judge3:
            index1 = sen.find('[math]')
            index2 = sen.replace('[math]', '      ', 1).find('[math]') + 6
            sen = sen.replace(sen[index1: index2], ' formula ')
        else:
            break
    return sen


def preprocessing(file_dir, rectify, train_or_dev=True):
    df = pd.DataFrame(pd.read_csv(file_dir, engine='python'))
    size = df.shape[0]
    question_text = df['question_text'].fillna('').apply(
        lambda x: remove_link_and_slash_split(remove_formula(x))).values
    question_text = [word_tokenize(line) for line in question_text]
    for i in range(size):
        for word in question_text[i]:
            if word in rectify:
                index = question_text[i].index(word)
                question_text[i][index] = rectify.get(word)
    df['question_text'] = question_text

    # [np.percentile(question_text_tokenized_len, 60 + 2 * p) for p in range(21)]
    # [14.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 17.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 25.0, 27.0, 29.0, 32.0, 39.0, 412.0]
    if train_or_dev:
        df = df.sample(frac=1.0)
        dev_df = df[:dev_size]
        train_df = df[dev_size:]
        df_target_1 = train_df[train_df['target'] == 1]
        for i in range(FLAGS.ratio_1_0-1):
            train_df = train_df.append(df_target_1)
        train_df = train_df.sample(frac=1.0)
    else:
        train_df = df
        dev_df = None
    return train_df, dev_df


def get_embedding_index(emb_dir):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embedding_index = dict(get_coefs(*line.split(" ")) for line in open(emb_dir))
    return embedding_index


def get_batch_data(train_df, batch_size, embedding_index, train_or_dev=True):
    if train_or_dev:
        train_df_batch = train_df.sample(n=batch_size)
    else:
        train_df_batch = train_df
    y = np.zeros([batch_size, 2])
    if train_or_dev:
        y_ = np.asarray(train_df_batch['target'])
        for i in range(batch_size):
            y[i, int(y_[i])] = 1
    x = train_df_batch['question_text'].values
    zeros = np.zeros([batch_size, FLAGS.q_max_len, word_vec_len])
    x_valid = np.zeros([batch_size], dtype=int)
    for n in range(batch_size):
        for word in x[n]:
            if word in embedding_index:
                zeros[n, x_valid[n], :] = embedding_index.get(word)
                x_valid[n] += 1
            # elif word in assign_to_variable:
            #     zeros[n, x_valid[n], :] += assign_to_variable.get(word)
            #     x_valid[n] += 1
            if x_valid[n] >= FLAGS.q_max_len:
                break
    x = zeros
    return x, x_valid, y


def get_cell(rnn_type, hidden_size, scope, layer_num, dp_keep_prob, training):
    """
    Gets the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        rnn_layer_num: MultiRNNCell are used if rnn_layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        cells: An RNN Cell
    """
    with tf.name_scope(scope):
        cells = []
        for i in range(layer_num):
            if rnn_type.endswith('lstm'):
                cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            elif rnn_type.endswith('gru'):
                cell = tc.rnn.GRUCell(num_units=hidden_size)
            elif rnn_type.endswith('rnn'):
                cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
            else:
                raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
            if training == True:
                cell = tc.rnn.DropoutWrapper(
                    cell, input_keep_prob=dp_keep_prob, output_keep_prob=dp_keep_prob)
            cells.append(cell)
        cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells

'''
def transformer_self_attention(inputs, att_keep_prob, is_train=True):
    if is_train:
        d_inputs = tf.nn.dropout(inputs, keep_prob=att_keep_prob,
                                 noise_shape=[inputs.get_shape()[0], FLAGS.q_max_len, 1])
        # noise_shape: determine which axis of elements to be dropout consistently.
    else:
        d_inputs = inputs

    inputs_ = tf.nn.relu(tf.layers.dense(d_inputs, rnn_hidden_size, use_bias=True))
    outputs = tf.matmul(inputs_, tf.transpose(inputs_, [0, 2, 1])) / (rnn_hidden_size ** 0.5)
    # zeros = 0.0 * outputs
    zeros = tf.zeros_like(outputs)
    att_weights = tf.nn.softmax(tf.where(~tf.equal(outputs, zeros), outputs, zeros-float('inf')))
    outputs = tf.matmul(att_weights, inputs)
    res = tf.concat([inputs, outputs], axis=-1)
    # res = inputs + outputs
    return res
'''


def bn(inputs, scope, epsilon=1e-5):
    with tf.variable_scope(scope):
        m, v = tf.nn.moments(inputs, -1, keep_dims=True)
        offset = tf.get_variable('offset', [1], tf.float32, tf.constant_initializer(0.0))
        scale = tf.get_variable('scale', [1], tf.float32, tf.constant_initializer(1.0))
        outputs = tf.nn.batch_normalization(inputs, m, v, offset, scale, epsilon)
    return outputs


def score(logits, targets):
    '''
        logits: [None, 2]
        targets: [None, 2]
    '''
    pred = tf.argmax(logits, -1)
    actual = tf.argmax(targets, -1)
    TP = tf.count_nonzero(pred * actual)
    TN = tf.count_nonzero((pred - 1) * (actual - 1))
    FP = tf.count_nonzero(pred * (actual - 1))
    FN = tf.count_nonzero((pred - 1) * actual)
    recall = tf.divide(TP, TP+FN)
    precision = tf.divide(TP, TP+FP)
    accu = tf.divide(TP+TN, TP+TN+FP+FN)
    f1 = tf.divide(2*recall*precision, recall+precision)
    return recall, precision, accu, f1


def main(_):
    # Building the whole model structure, training and testing.
    with tf.variable_scope('prepare_data'):
        # thisisaformula = tf.get_variable('thisisaformula', [word_vec_len], tf.float32,
        #                                  tf.contrib.layers.xavier_initializer()),
        # thisisalink = tf.get_variable('thisisalink', [word_vec_len],
        #                               tf.float32, tf.contrib.layers.xavier_initializer())
        # assign_to_variable_dict = {'thisisaformula': thisisaformula, 'thisisalink': thisisalink}
        train_states = tf.placeholder(dtype=tf.bool)
        X = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.q_max_len, word_vec_len], name='X')
        X_valid = tf.placeholder(dtype=tf.int32, shape=[None], name='X_valid')
        Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='Y')
        train_df, dev_df = preprocessing(train_dir, rectify_dict)
        embedding_index = get_embedding_index(embedding_dir)

    '''
    with tf.variable_scope('BiRNN_encoder'):
        rnn_cell_fw = get_cell('gru', rnn_hidden_size, 'rnn_cell_fw',
                               rnn_layer_num, dp_keep_prob=FLAGS.dp_keep_prob, training=train_states)
        rnn_cell_bw = get_cell('gru', rnn_hidden_size, 'rnn_cell_bw',
                               rnn_layer_num, dp_keep_prob=FLAGS.dp_keep_prob, training=train_states)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            rnn_cell_fw, rnn_cell_bw, X, sequence_length=X_valid, dtype=tf.float32)
        # outputs:([None, FLAGS.q_max_len, rnn_hidden_size], [None, FLAGS.q_max_len, rnn_hidden_size])
        # states:((fw[None, rnn_hidden_size]*2), (bw[None, rnn_hidden_size]*2))
        # outputs = tf.concat(outputs, 2)  # [None, FLAGS.q_max_len, 2*rnn_hidden_size]
        useful_states = tf.concat((states[0][-1], states[1][-1]), -1)  # [None, 2*rnn_hidden_size]
        
    '''

    with tf.variable_scope('CNN_encoder'):
        filter_sizes = [3, 4, 5]
        num_filters = 2
        pooled_outputs = []
        x_image = tf.reshape(X, [-1, FLAGS.q_max_len, word_vec_len, 1])
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, word_vec_len, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(x_image, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, FLAGS.q_max_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # dropout
        h_drop = tf.nn.dropout(h_pool_flat, FLAGS.dp_keep_prob_train)

    '''
    with tf.variable_scope('self_attention'):
        # Self Attention in Transformer with concat.
        fusion = transformer_self_attention(outputs, att_keep_prob)  # [None, FLAGS.q_max_len, 4*rnn_hidden_size]
        fusion_rnn_fw = get_cell('gru', rnn_hidden_size, 'fusion_rnn_fw', layer_num=1,
                                 dropout_keep_prob=rnn_dropout_keep_prob, is_train=True)
        fusion_rnn_bw = get_cell('gru', rnn_hidden_size, 'fusion_rnn_bw', layer_num=1,
                                 dropout_keep_prob=rnn_dropout_keep_prob, is_train=True)
        _, fusion_final = tf.nn.bidirectional_dynamic_rnn(
            fusion_rnn_fw, fusion_rnn_bw, fusion, sequence_length=X_valid, dtype=tf.float32)
        fusion_final_states = tf.concat((fusion_final[0][0], fusion_final[1][0]), -1)
        # [None, 2*rnn_hidden_size]
    '''

    with tf.variable_scope('MLP'):
        # MLP with BN

        # layer1 = bn(useful_states, 'MLP_layer1') # use BiLSTM
        layer1 = bn(h_drop, 'MLP_layer1') # use CNN
        layer1_d = tf.layers.dropout(layer1, 1-FLAGS.dp_keep_prob, training=train_states)
        logits2 = tf.layers.dense(layer1_d, MLP_hidden_layer, kernel_initializer=tf.truncated_normal_initializer())
        logits2_bn = bn(logits2, 'MLP_layer2')
        layer2 = tf.nn.relu(logits2_bn)
        layer2_d = tf.layers.dropout(layer2, 1-FLAGS.dp_keep_prob, training=train_states)
        logits3 = tf.layers.dense(layer2_d, 2, kernel_initializer=tf.truncated_normal_initializer())
        # [[p_0, p_1],...,[p_0,p_1]]
        softmax_outputs = tf.nn.softmax(logits3)
        target_pred = tf.argmax(logits3, -1)

    recall, precision, accu, f1 = score(logits=logits3, targets=Y)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(Y), logits=logits3))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    with tf.name_scope('summaries'):
        train_recall = tf.summary.scalar('train_recall', recall)
        train_precision = tf.summary.scalar('train_precision', precision)
        train_f1 = tf.summary.scalar('train_f1', f1)
        train_loss = tf.summary.scalar('train_loss', loss)
        dev_recall = tf.summary.scalar('dev_recall', recall)
        dev_precision = tf.summary.scalar('dev_precision', precision)
        dev_f1 = tf.summary.scalar('dev_f1', f1)
        dev_loss = tf.summary.scalar('dev_loss', loss)

    merged_train = tf.summary.merge([train_recall, train_precision, train_f1, train_loss])
    merged_dev = tf.summary.merge([dev_recall, dev_precision, dev_f1, dev_loss])
    time_before_sess = time.time()
    print('Now build session.')

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(tensorboard_log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(n_iteration):
            x, x_valid, y = get_batch_data(train_df, batch_size, embedding_index)
            _ = sess.run(optimizer,
                         feed_dict={X: x, X_valid: x_valid, Y: y, train_states: True})
            if (i+1) % 100 == 0:
                x_dev, x_valid_dev, y_dev = get_batch_data(
                    dev_df, dev_size, embedding_index)
                merged_train_ = sess.run(merged_train, feed_dict={
                    X: x, X_valid: x_valid, Y: y, train_states: False})
                summary_writer.add_summary(merged_train_, i+1)
                merged_dev_ = sess.run(merged_dev, feed_dict={
                    X: x_dev, X_valid: x_valid_dev, Y: y_dev, train_states: False})
                summary_writer.add_summary(merged_dev_, i + 1)
        summary_writer.close()
        print('time used before sess is {0}'.format(time_before_sess - start_time))
        print(tensorboard_log_dir+' training completed.')
        print('training time used= {0}'.format(time.time()-time_before_sess))

        # now predict for the test set.
        # df_test, _ = preprocessing(test_dir, False)
        # test_size = df_test.shape[0]
        # prediction = []
        # for i in range(int(1e10)):
        #     df = df_test.iloc[i*100:(i+1)*100]
        #     x_test, x_valid_test, y_test = get_batch_data(df, df.shape[0], embedding_index, False)
        #     predict_test = sess.run(target_pred, feed_dict={X: x_test, X_valid: x_valid_test, train_states: False})
        #     prediction += list(predict_test)
        #     if (i+1)*100 >= test_size:
        #         break
        # df_test['prediction'] = prediction
        # submission = df_test.drop('question_text', axis=1)
        # submission.to_csv('./data/submission.csv', index=False)
        # print('write to submission.csv successfully')


if __name__ == '__main__':
    tf.app.run()
