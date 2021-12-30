import string
import numpy as np
import re, os, string
from pyvi import ViTokenizer
from pyvi.ViTokenizer import tokenize
import tensorflow as tf
from gensim.models.fasttext import FastText 
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import *
from gensim.models import KeyedVectors
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import pickle
# from fairseq.data.encoders.fastbpe import fastBPE
# from fairseq.data import Dictionary
import argparse
MAX_LEN = 125

###### Preprocessing Text ######
# Loại bỏ các ký tự thừa
def clean_text(text):
    # text = re.sub(r"^A-Za-z0-9", "", text.decode('utf-8'))
    text = re.sub(r"^A-Za-z0-9", "", text)
    text = text.strip()
    return text

#tách câu
def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents

#tách từ
def word_segment(sent):
    sent = tokenize(sent)
    return sent

#Chuẩn hóa từ
def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()


with open("./vietnamese-stopwords.txt",encoding="utf8") as f:
    list_stopwords = f.readlines()
def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
        text2 = ' '.join(pre_text)
    return text2

#### embedding and padding ####
max_length_inp = 125
def sentence_embedding(sent,fast_text_model):
    # content = preprocess(sent)
    content = clean_text(sent) 
    content = word_segment(content)
    content = remove_stopword(normalize_text(content))
    inputs = []
    for word in content.split():
      if word in fast_text_model.vocab:
        inputs.append(fast_text_model.wv.get_vector(word))
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=max_length_inp,
                                                                dtype='float32',
                                                                padding='post')
    return inputs

def predict_intent(input_sent, model, fast_text_model):
    embed_sent = sentence_embedding(input_sent, fast_text_model)[0]
    predict_x = model.predict(np.expand_dims(embed_sent, 0)) 
    predict_class=np.argmax(predict_x,axis=1)
    return predict_class

def loadModel(link_to_model):
    print("Loading model....")
    model =  tf.keras.models.load_model(link_to_model)
    print("Model loaded succesfully")
    return model

def loadFastText():
    #implement fastext-model
    print("Load FastText ....")
    fast_text_model = KeyedVectors.load('./weights/fasttext_gensim.model')
    print("Load FastText successfully")
    return fast_text_model

# def loadBPE():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--bpe-codes', 
#         default="./weights/PhoBERT_base_transformers/bpe.codes",
#         required=False,
#         type=str,
#         help='path to fastBPE BPE'
#     )
#     args, unknown = parser.parse_known_args()
#     bpe = fastBPE(args)
#     print("Loading BPE succesfully")
#     return bpe

def load_model_torch():
    model = torch.load("./weights/pytorch_entire_ID_labeled.pt",map_location=torch.device('cpu'))
    print("Load model Torch sucessfully")
    return model

def load_vocab():
    vocab = Dictionary()
    vocab.add_from_file("./weights/PhoBERT_base_transformers/dict.txt")
    print("Load vocab sucessfully")
    return vocab

def predict_intent_pytorch(sent, bpe, vocab, model):
    content = clean_text(sent) 
    content = word_segment(content)
    seed = remove_stopword(normalize_text(content))
    subwords = '<s> ' + bpe.encode(seed) + ' </s>'
    encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()

    pad_sent = padding(encoded_sent)
    mask = [int(token_id > 0) for token_id in pad_sent]

    model.eval()
    pad_sent = torch.tensor(pad_sent)
    mask = torch.tensor(mask)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pad_sent = pad_sent.to(device)
    mask = mask.to(device)
    outputs = model(pad_sent.unsqueeze(0),  token_type_ids=None, attention_mask=mask.unsqueeze(0))
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    # print('values',logits[0][0])
    # print(np.argmax(logits, axis=1))
    # print('Label',le.classes_[np.argmax(logits, axis=1)])
    classes = ['dat_bao_thuc', 'dat_nha_hang', 'lich_lam_viec', 'mo_nhac','so_du_tai_khoan']
    return classes[np.argmax(logits[0])]
    
def padding(encoded_sent):
    x = []
    input_ids = encoded_sent
    input_ids = input_ids[:min(len(input_ids), MAX_LEN - 2)]
    input_ids = input_ids + [0] * (MAX_LEN - len(input_ids))
    x.append(np.array(input_ids))
    return (np.array(x).reshape(125))

def load_tfidf_vectorizer():
    file = open('./weights/vectorizer.pk', 'rb')
    tfidf_pickle = pickle.load(file)
    print("Load TFIDF vectorizer succesfully")
    return tfidf_pickle

def sentence_processing(text):
    content = clean_text(text) 
    content = word_segment(content)
    content = remove_stopword(normalize_text(content))
    return content

def predict_intent_tfidf(input_sent,tfidf_pickle, model):
    intents = ['so_du_tai_khoan', 'mo_nhac', 'lich_lam_viec', 'dat_nha_hang', 'dat_bao_thuc']
    embed_sent = tfidf_pickle.transform([sentence_processing(input_sent)])
    predict_x = model.predict(embed_sent.todense()) 
    predict_class=np.argmax(predict_x,axis=1)
    return intents[predict_class[0]]

if __name__ == "__main__":
    pass


