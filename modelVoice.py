import glob
import torch
import time
import torchaudio
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import torchaudio
import torchaudio.sox_effects as S
import torchaudio.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
import os

class TDNN(nn.Module):
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=False,
                    dropout_p=0.2
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, input_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x

class X_vector(nn.Module):
    def __init__(self, input_dim = 40, num_classes = 26):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=256, context_size=3, dilation=2,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=256, output_dim=256, context_size=3, dilation=3,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=256, output_dim=256, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=256, output_dim=256, context_size=1, dilation=1,dropout_p=0.5)
        self.segment6 = nn.Linear(512, 256)
        self.segment7 = nn.Linear(256, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)

        mean = torch.mean(tdnn5_out,1)
        std = torch.var(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)

        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)

        predictions = self.output(x_vec)

        return predictions, x_vec

def wav2mfccs_test(path, useVAD = True):
  # converM4A_WAV(path)
  y, sr = torchaudio.sox_effects.apply_effects_file(path, effects=[['rate','16000'],['channels','1']])
  print(len(y[0])/sr)
  ts = get_speech_ts(y[0], model_vad, sampling_rate=16000)
  if useVAD:
    if len(ts)>0:
      y = torch.cat([y[:,i['start']:i['end']] for i in ts], axis=1)
    else:
      return None
  mfccs = torch.cat([mfcc_transform(y[:,i:i+int(window_size*sr)]) for i in range(0,int(y.shape[1])-int(window_size*sr)+1,int(window_step*sr))])
  mfccs = torch.transpose(mfccs,1,2)
  return mfccs

def extract_Xvector(pathAudio):
  with torch.no_grad():
    mfccs = wav2mfccs_test(pathAudio)
    if mfccs == None:
      return None
    features = mfccs.to(device)
    _, xvector = model(features)

  return xvector

data = np.load("./weights/dataEmbedding.npy", allow_pickle=True)
#@title Xvector
sample_rate = 16000
n_mfcc = 40
window_size = 0.25
window_step = 0.05

num_frame = 21
num_input = 40
num_classes = 46

#Khoi tao mo hinh
path_pretrain = "./weights/pretrain.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = X_vector(num_input, num_classes).to(device)
if torch.cuda.is_available():
  model.load_state_dict(torch.load(path_pretrain)['model'])
else:
  model.load_state_dict(torch.load(path_pretrain,map_location=torch.device('cpu'))['model'])
model.eval()

# Khoi tao embedding+xoa khoang lang
# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_ts,_, read_audio,*_) = utils
mfcc_transform = T.MFCC(sample_rate=sample_rate,n_mfcc=n_mfcc)

#khoi tao mo hinh tranning
data_spk_2 = dict()
for spk,lst in data[()]['train'].items():
  lst_spk = []
  i = 0 
  for utt,vec in lst.items():
    lst_spk.append(vec)
    i+=1
    if(i > 5):
      break
  data_spk_2[spk] = np.vstack(lst_spk)


#mean data
mean_data_spk = dict()
for spk,lst in data_spk_2.items():
  mean_per_spk = np.mean(data_spk_2[spk], axis=0)
  mean_data_spk[spk] = mean_per_spk


def addData2Mean(your_data, your_name):
  mean_own_data = np.mean(your_data, axis=0)
  mean_data_spk[your_name] = mean_own_data
  print(list(mean_data_spk.keys()))

def findCosineDistance(source_representation, test_representation):
  a = np.matmul(np.transpose(source_representation), test_representation)
  b = np.sum(np.multiply(source_representation, source_representation))
  c = np.sum(np.multiply(test_representation, test_representation))
  return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findLabelOfVoice(input_data, dict_data, threshold):
  distance = []
  for spk in dict_data.keys():
    dist = findCosineDistance(np.mean(input_data,axis=0), dict_data[spk])
    distance.append(dist)
  print(distance)
  #find element < threshold
  elements = list(filter(lambda i: i < threshold, distance))
  print(elements)
  labels = list(dict_data.keys())
  if len(elements) == 0:
    return 'None'
  if len(elements)>1:
    return labels[distance.index(np.min(distance))]
  return labels[distance.index(elements[0])]

def predictVoice(data_test):
  #check label of mean data
  print("Label of data", mean_data_spk.keys())
  label = findLabelOfVoice(data_test, mean_data_spk, threshold=0.05)
  return label

def loadAudioFolder(dir_audio):
    data_audio = []
    for file in os.listdir(dir_audio):
        filepath = os.path.join(dir_audio, file)
        xvector = extract_Xvector(filepath)
        data_audio.append(xvector)
    final_data = np.vstack(data_audio)
    print("Check shape data", final_data.shape)
    return final_data



