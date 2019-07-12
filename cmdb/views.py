from django.shortcuts import render
from django.shortcuts import HttpResponse

from SegNer.CRF_Segment import prediction_random as crfseg
from HMM_Segment import HMM as hmmseg
from CRF_NER import prediction_random as crfner
from Seg2Html import seg2html,ner2html,Bner2html
from BiLSTM_Segment import predict_random as bicrfseg
from BiLSTM_NER import predict_random as bicefner


def index(request):

    return render(request,'index.html')

def demo_ajax(request):
    return render(request, 'index2.html')


def index_show(request):
    a=request.GET['input1']
    b=request.GET['mood']


    if request.is_ajax():
        ajax_string = 'ajax request: '
    else:
        ajax_string = 'not ajax request: '

    r = HttpResponse(ajax_string+"23213123213")
    return r


def demo_add(request):
    input1 = request.GET['input1']
    mood = request.GET['mood']

    if mood == 'crf':
        res = seg2html(crfseg(sent=input1))

    elif mood == 'hmm':
        res = seg2html(hmmseg(sequence=input1))

    elif mood == 'crf2':
        sent = crfseg(sent=input1)
        ner = crfner(sent)
        res = ner2html(sent,ner)

    elif mood == 'BiLSTM1':
        res =seg2html(bicrfseg(demo_sent=input1))

    elif mood == 'BiLSTM2':
        sent,label = bicefner(sent=input1)
        res = Bner2html(sent=sent,ner=label)
    else:
        res = ""

    r = HttpResponse(res)
    return r
# Create your views here.
