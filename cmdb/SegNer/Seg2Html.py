
from WordNERDic import WordDic

# <div class="n0">吴金明/nr</div>
#分词类型
classseg ="n19"



bner = {
    'ORG':'n14',
    'PER':'n21',
    'LOC':'n15'
}

def seg2html(sent):
    res = ""
    for item in sent:
        it1 = '<div class="'+classseg+'">'+item+'</div>'
        res+=it1
    return res


def ner2html(sent,ner):
    res = ""
    for i in range(len((sent))):
        it1 = '<div class="'+str(WordDic.setdefault(ner[i],'n22'))+'">'+sent[i]+'</div>'
        res+=it1
    return res

def Bner2html(sent,ner):
    res = ""
    for i in range(len((sent))):
        it1 = '<div class="'+str(bner.setdefault(ner[i],'n19'))+'">'+sent[i]+'</div>'
        res+=it1
    return res


if __name__ == '__main__':
    sent = ['一般来说','，','监督','学习','可以','看','做','最小','化','下面','的','目标','函数']
    nr = ['l', 'w', 'v', 'v', 'v', 'v', 'v', 'n', 'v', 'f', 'u', 'n', 'n']
    temp = ['不要', '有', '列表', '的', '嵌套', '的', '形式']
    print(ner2html(sent,nr))
