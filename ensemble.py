import tensorflow as tf
import tensorflow.contrib as tc
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

batch_size = 512
q_max_len = 32
n_iteration_list = [7300, 7000, 9000]
word_vec_len = 300
rnn_hidden_size = 150
rnn_layer_num = 2
MLP_hidden_layer = 150
dp_keep_prob_train_list = [0.2, 0.5, 0.8]
ratio_1_0 = 2
dev_size = 20000
n_train = 3

train_dir = '../input/train.csv'
test_dir = '../input/test.csv'
embedding_dir = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'


def preprocessing():
    train_df = pd.DataFrame(pd.read_csv(train_dir, engine='python'))
    train_df['question_text'] = [word_tokenize(line) for line in train_df['question_text'].fillna('').values]
    train_df = train_df.sample(frac=1.0)
    dev_df = train_df[:dev_size]
    train_df = train_df[dev_size:]
    df_target_1 = train_df[train_df['target'] == 1]
    for i in range(ratio_1_0 - 1):
        train_df = train_df.append(df_target_1)
    train_df = train_df.sample(frac=1.0)
    return train_df, dev_df


def get_embedding_index(emb_dir):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

    embedding_index = dict(get_coefs(*line.split(" ")) for line in open(emb_dir))
    return embedding_index


def get_batch_data(train_df, batch_size, embedding_index, train_or_dev=True):
    if train_or_dev == True:
        train_df_batch = train_df.sample(n=batch_size)
    else:
        train_df_batch = train_df
    y = np.zeros([batch_size, 2])
    if train_or_dev:
        y_ = np.asarray(train_df_batch['target'])
        for i in range(batch_size):
            y[i, int(y_[i])] = 1
    x = train_df_batch['question_text'].values
    zeros = np.zeros([batch_size, q_max_len, word_vec_len])
    x_valid = np.zeros([batch_size], dtype=int)
    for n in range(batch_size):
        for word in x[n]:
            if word in embedding_index:
                zeros[n, x_valid[n], :] = embedding_index.get(word)
                x_valid[n] += 1
            if x_valid[n] >= q_max_len:
                break
    x = zeros
    return x, x_valid, y


def get_cell(rnn_type, hidden_size, scope, layer_num, dp_keep_prob, training):
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


def bn(inputs, scope, epsilon=1e-5):
    with tf.variable_scope(scope):
        m, v = tf.nn.moments(inputs, -1, keep_dims=True)
        offset = tf.get_variable('offset', [1], tf.float32, tf.constant_initializer(0.0))
        scale = tf.get_variable('scale', [1], tf.float32, tf.constant_initializer(1.0))
        outputs = tf.nn.batch_normalization(inputs, m, v, offset, scale, epsilon)
    return outputs


def softmax_to_pred(softmax, thereshold):
    pred = []
    len = softmax.shape[0]
    for i in range(len):
        if softmax[i, 1] >= thereshold:
            pred.append(1)
        else:
            pred.append(0)
    return pred


def model(nth_train, train, embedding, test, test_num, dp_prob, iteration):
    # Building the whole model structure, training and testing.
    with tf.variable_scope('prepare_data' + str(nth_train)):
        train_states = tf.placeholder(dtype=tf.bool)
        X = tf.placeholder(dtype=tf.float32, shape=[None, q_max_len, word_vec_len], name='X')
        X_valid = tf.placeholder(dtype=tf.int32, shape=[None], name='X_valid')
        Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='Y')

    with tf.variable_scope('BiRNN_encoder' + str(nth_train)):
        rnn_cell_fw = get_cell('gru', rnn_hidden_size, 'rnn_cell_fw',
                               rnn_layer_num, dp_keep_prob=dp_prob, training=train_states)
        rnn_cell_bw = get_cell('gru', rnn_hidden_size, 'rnn_cell_bw',
                               rnn_layer_num, dp_keep_prob=dp_prob, training=train_states)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            rnn_cell_fw, rnn_cell_bw, X, sequence_length=X_valid, dtype=tf.float32)
        states = tf.concat((states[0][-1], states[1][-1]), -1)  # [None, 2*rnn_hidden_size]

    with tf.variable_scope('MLP' + str(nth_train)):
        layer1 = bn(states, 'MLP_layer1')
        layer1_d = tf.layers.dropout(layer1, 1 - dp_prob, training=train_states)
        logits2 = tf.layers.dense(layer1_d, MLP_hidden_layer, kernel_initializer=tf.truncated_normal_initializer())
        logits2_bn = bn(logits2, 'MLP_layer2')
        layer2 = tf.nn.relu(logits2_bn)
        layer2_d = tf.layers.dropout(layer2, 1 - dp_prob, training=train_states)
        logits3 = tf.layers.dense(layer2_d, 2, kernel_initializer=tf.truncated_normal_initializer())
        # [[p_0, p_1],...,[p_0,p_1]]
        softmax_outputs = tf.nn.softmax(logits3)
        target_pred = tf.argmax(logits3, -1)

    with tf.variable_scope('eval' + str(nth_train)):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(Y), logits=logits3))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    print('now bulid session ' + str(nth_train))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iteration):
            x, x_valid, y = get_batch_data(train, batch_size, embedding)
            _ = sess.run(optimizer,
                         feed_dict={X: x, X_valid: x_valid, Y: y, train_states: True})
        print('training done.')

        # now predict for the test set.
        softmax_prediction = np.asarray([0.0, 0.0], dtype=np.float32)
        for i in range(int(1e10)):
            df = test.iloc[i * 100:(i + 1) * 100]
            x_test, x_valid_test, y_test = get_batch_data(df, int(df.shape[0]), embedding, False)
            softmax_outputs_ = sess.run(softmax_outputs,
                                        feed_dict={X: x_test, X_valid: x_valid_test, train_states: False})
            softmax_prediction = np.vstack((softmax_prediction, softmax_outputs_))
            if (i + 1) * 100 >= test_num:
                break
        softmax_prediction = softmax_prediction[1:]
    return softmax_prediction


def main():
    train_df = preprocessing()
    embedding_index = get_embedding_index(embedding_dir)

    df_test = pd.DataFrame(pd.read_csv(test_dir, engine='python'))
    df_test['question_text'] = [word_tokenize(line) for line in df_test['question_text'].fillna('').values]
    test_size = df_test.shape[0]

    prediction = np.zeros([test_size, 2])
    for n in range(n_train):
        n_iteration = n_iteration_list[n]
        dp_keep_prob_train = dp_keep_prob_train_list[n]
        softmax_prediction = model(n, train_df, embedding_index, df_test, test_size, dp_keep_prob_train, n_iteration)
        prediction += softmax_prediction
    prediction /= n_train

    pred_test = softmax_to_pred(prediction, 0.5)
    df_test['prediction'] = pred_test
    submission = df_test.drop('question_text', axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
