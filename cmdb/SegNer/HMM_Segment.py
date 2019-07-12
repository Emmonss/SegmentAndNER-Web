import numpy as np
import math
import pickle
from AbsolutePath import HmmDIC,HmmDISTRIBUTION


def Label2Tag(label):
    tag = { 0:'B',1:'M',2:'E',3:'S'}
    return tag[label]

def LoadMatrix():
    with open(HmmDISTRIBUTION, 'rb') as fr:
        ProDic=pickle.load(fr)

    with open(HmmDIC, 'rb') as fr:
        Dic=pickle.load(fr)

    return Dic,ProDic

def getMaxPre(prob,Dic,ProDic):
    nextprob,maxpre = [],[]
    for i in range(len(Dic)):
        problist = []
        for j in range(len(prob)):
            problist.append(math.sqrt(prob[j]*Dic[i]*ProDic[j][i]))
        nextprob.append(max(problist))
        maxpre.append(problist.index(max(problist)))

    return nextprob,maxpre

def wordpred(word,Dic):
    if word in Dic:
        return Dic[word]
    else:
        return [0.0,0.0,0.0,1.0]

def Forward(sequence, Dic, ProDic):
    prob = np.zeros((len(sequence),4))
    maxpre = np.zeros((len(sequence)-1,4))
    prob[0] = wordpred(sequence[0],Dic)
    for i in range(1,len(sequence)):
        prob[i],maxpre[i-1] = getMaxPre(prob[i-1],wordpred(sequence[i],Dic),ProDic)

    lastmaxpro = prob[-1].tolist().index(max(prob[-1]))
    return lastmaxpro,maxpre

def Backward(lastmaxpro,maxpre):
    tag = []
    tag.append(Label2Tag(lastmaxpro))
    for i in range(len(maxpre)-1,-1,-1):
        lastmaxpro =(int)(maxpre[i].tolist()[lastmaxpro])
        tag.append(Label2Tag(lastmaxpro))
    tag.reverse()
    res = "".join(tag)
    return res


def segment(sent,tag):
    res = ''
    for i in range(len(sent)):
        if tag[i] == 'S' or tag[i] == 'E':
            res+=sent[i]+" "
        else:
            res+=sent[i]
    return res

def HMM(sequence):
    Dic, ProDic = LoadMatrix()
    lastmaxpro, maxpre = Forward(sequence, Dic=Dic, ProDic= ProDic)
    tag = Backward(lastmaxpro,maxpre)
    res = segment(sequence,tag)
    return res.split()

if __name__ == '__main__':
    sequence = '希腊的经济结构较特殊。'
    res = HMM(sequence)
    print(res)