import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """
    BUG: I forget to transform all the English characters from full-width into half-width... 
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    从路径文件中读取字典
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data: 数据集
    :param batch_size: batch大小
    :param vocab: 词典
    :param tag2label:标签对序号BMES===>0123
    :param shuffle: 是否打乱数据
    :return:
    """

    #打乱数据
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    ##返回batch
    for (sent_, tag_) in data:
        #字对应序号
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

# word2id = read_dictionary('./data_path/word2id.pkl')
# train = read_corpus("./data_path/test_data")
# print(np.shape(train))
# batches = batch_yield(train, batch_size=10, vocab=word2id,tag2label=tag2label,shuffle=True)
# for step, batch in enumerate(batches):
#     print(np.shape(batch))
#     print(batch[0])
#     print(batch[1])
#     break
# print(np.shape(batches))

# train = read_corpus("./data_path/test_data")
# print(train)
# word2id = read_dictionary('./data_path/word2id.pkl')
# batches = batch_yield(train, 3, word2id, tag2label, shuffle=True)
# i=0
# for seq,labels in batches:
#     i+=1
#     print(seq)
#     print(labels)
#     #print(len(batches))
#     print("+++++++++++++++++++++")
# print(i)
#data = load_data(flag.train_data_path)
# word2id = read_dictionary('./data_path/word2id.pkl')
# emd = random_embedding(word2id ,4)
# # data = vocab_build("da",flag.train_data_path,1)
#print(emd)
# word2id = read_corpus("./data_path/test_data")
# print(word2id)
#emd = random_embedding(word2id , 300)
#print(emd)