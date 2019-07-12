import pycrfsuite
from AbsolutePath import CrfSegMoodPath


def segment(sent,tag):
    res = ''
    for i in range(len(sent)):
        if tag[i] == 'S' or tag[i] == 'E':
            res+=sent[i]+" "
        else:
            res+=sent[i]
    return res.split()


def Word2Features(sent,i):
    word = sent[i][0]
    features = [
        'bias',
        'word=' + word,
        'word.islower={}'.format(word.islower()),
        'word.isupper={}'.format(word.isupper()),
        'word.isdigit={}'.format(word.isdigit()),
    ]

    if i > 0:
        word1 = sent[i-1][0]
        words = word1+word
        features.extend([
            '-1:word=' + word1,
            '-1:words=' + words,
            '-1:word.isupper={}'.format(word1.isupper()),
            '-1:word.isdigit={}'.format(word.isdigit()),
        ])
    else:
        features.append('BOS')

    if i>1:
        word1 = sent[i - 2][0]
        word2 = sent[i - 1][0]
        words = word1+word2+word
        features.extend([
            '-2:word=' + word1,
            '-2:words=' + words,
            '-2:word.isupper={}'.format(word1.isupper()),
            '-2:word.isdigit={}'.format(word.isdigit()),
        ])

    if i > 2:
        word1 = sent[i - 3][0]
        word2 = sent[i - 2][0]
        word3 = sent[i - 1][0]
        words = word1 + word2 + word3 + word
        features.extend([
            '-3:word=' + word1,
            '-3:words=' + words,
            '-3:word.isupper={}'.format(word1.isupper()),
            '-3:word.isdigit={}'.format(word.isdigit()),
        ])



    if i < len(sent)-1:
        word1 = sent[i + 1][0]
        words = word +  word1
        features.extend([
            '+1:word=' + word1,
            '+1:words=' + words,
            '+1:word.isupper={}'.format(word1.isupper()),
            '+1:word.isdigit={}'.format(word.isdigit()),
        ])
    else:
        features.append('EOS')


    if i < len(sent)-2:
        word1 = sent[i + 2][0]
        word2 = sent[i + 1][0]
        words = word +  word1 + word2
        features.extend([
            '+2:word=' + word1,
            '+2:words=' + words,
            '+2:word.isupper={}'.format(word1.isupper()),
            '+2:word.isdigit={}'.format(word.isdigit()),
        ])


    if i < len(sent)-3:
        word1 = sent[i + 3][0]
        word2 = sent[i + 2][0]
        word3 = sent[i + 1][0]
        words = word +  word1 + word2 + word3
        features.extend([
            '+3:word=' + word1,
            '+3:words=' + words,
            '+3:word.isupper={}'.format(word1.isupper()),
            '+3:word.isdigit={}'.format(word.isdigit()),
        ])

    return features

def sent2features(sent):
    return [Word2Features(sent, i) for i in range(len(sent))]

def MakeOneNode(sent):
    res = []
    for word in sent:
        res.append(word)
    return res


def prediction_random(sent,moodpath = CrfSegMoodPath):
    example = MakeOneNode(sent)

    tagger = pycrfsuite.Tagger()
    tagger.open(moodpath)
    res = segment(example,tagger.tag(sent2features(example)))
    return res


if __name__ == '__main__':
    # prediction()
    res = prediction_random("不要有列表的嵌套的形式")
    print(res)