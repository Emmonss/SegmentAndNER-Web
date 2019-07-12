import tensorflow as tf
import os
from BiLSTMNER.NER import BiLSTM_CRF
from BiLSTMNER.data import read_dictionary, tag2label, random_embedding
from AbsolutePath import BiLSTMNERPath
from Seg2Html import Bner2html

def getDicEmbed():
    word2id = read_dictionary(os.path.join(BiLSTMNERPath, 'word2id.pkl'))
    embeddings = random_embedding(word2id, 300)
    return word2id, embeddings


output_path = BiLSTMNERPath
summary_path = os.path.join(output_path, "summaries")
model_path = os.path.join(output_path, "checkpoints/")
ckpt_prefix = os.path.join(model_path, "model")
result_path = os.path.join(output_path, "results")
log_path = os.path.join(result_path, "log.txt")


def tag2will(sent,tag):
    res = ""
    label = []
    for i in range(len(tag)):
        if(tag[i]==0):
            res+=' '+sent[i]+" "
            label.append('0')
        else:
            res+=sent[i]
            temp = tag[i].split('-')[1]
            if(label==[]):
                label.append(temp)
            elif not (temp == label[len(label)-1]):
                label.append(temp)
    return res.split(),label

def predict_random(sent):
    ckpt_file = tf.train.latest_checkpoint(model_path)
    word2id, embeddings = getDicEmbed()
    tf.reset_default_graph()
    model = BiLSTM_CRF(batch_size=32, epoch_num=10, hidden_dim=300,
                       embeddings=embeddings,
                       dropout_keep=0.5, optimizer='Adam', lr=0.001, clip_grad=0.5,
                       tag2label=tag2label, vocab=word2id, shuffle=True,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=True, update_embedding=True)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})) as sess:
        saver.restore(sess, ckpt_file)
        demo_sent = list(sent.strip())
        demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        tag = model.demo_one(sess, demo_data)
        sent, label = tag2will(demo_sent,tag)
        sess.close()
    return sent,label


if __name__ == '__main__':
    sent = "国务院总理李克强在雄安新区召开会议"
    tag = ['B-ORG', 'I-ORG', 'I-ORG', 0, 0, 'B-PER', 'I-PER', 'I-PER', 0, 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 0, 0, 0, 0]
    sent,label = predict_random(sent)
    res = Bner2html(sent,label)
    print(res)