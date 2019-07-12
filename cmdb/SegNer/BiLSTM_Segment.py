import tensorflow as tf
import os
from BiLSTMSeg.BiLSTM_Model import BiLSTM_CRF
from BiLSTMSeg.data import read_dictionary,random_embedding
from AbsolutePath import BiLSTMCXPath

tag2label = {"B": 0,
             "M": 1,
             "E": 2,
             "S": 3,
             }

output_path = BiLSTMCXPath
summary_path = os.path.join(output_path, "summaries")
model_path = os.path.join(output_path, "checkpoints/")
ckpt_prefix = os.path.join(model_path, "model")
result_path = os.path.join(output_path, "results")
log_path = os.path.join(result_path, "log.txt")


def getDicEmbed():
    word2id = read_dictionary(os.path.join(BiLSTMCXPath, 'bilstm_word2id.pkl'))
    embeddings = random_embedding(word2id, 300)
    return word2id, embeddings

def segment(sent,tag):
    res = ''
    for i in range(len(sent)):
        if tag[i] == 'S' or tag[i] == 'E':
            res+=sent[i]+" "
        else:
            res+=sent[i]
    return res.split()



def predict_random(demo_sent):
    word2id, embeddings = getDicEmbed()
    ckpt_file = tf.train.latest_checkpoint(model_path)
    tf.reset_default_graph()
    model = BiLSTM_CRF(batch_size=32, epoch_num=10, hidden_dim=100,
                       embeddings=embeddings,
                       dropout_keep=0.5, optimizer='Adam', lr=0.001, clip_grad=5.0,
                       tag2label=tag2label, vocab=word2id, shuffle=True,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=True, update_embedding=True)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})) as sess:
        saver.restore(sess, ckpt_file)
        demo_sent = list(demo_sent.strip())
        demo_data = [(demo_sent, ['M'] * len(demo_sent))]
        tag = model.demo_one(sess, demo_data)
        sess.close()
    res = segment(demo_sent,tag)
    return res


if __name__ == '__main__':
    sent = "二者这个东西也不好讲清楚，尽管网上已经有朋友写得不错了(见文末参考链接)"
    print(predict_random(sent))