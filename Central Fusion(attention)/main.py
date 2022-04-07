# Code Author: Guanting Cen, Chongyang Wang (My PhD was supported by UCL Overseas Research Scholarship and Graduate Research Scholarship, 
# and partially by the EU Future and Emerging Technologies (FET) Proactive Programme H2020-EU.1.2.2 (Grant agreement 824160; EnTimeMent).

# Revision Date: 26 12 2021


import numpy as np
import pandas as pd
from collections import Counter
import h5py
import hdf5storage as hd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # enable a clean command line  window. The thing is to define os. before importing tensorflow.
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # use GPU with ID=0.
import tensorflow as tf
import keras
from tensorflow.keras.layers import * # for the new versions of Tensorflow, layers, models, regularizers, and optimizers shall be imported from Tensorflow.
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
from keras.losses import * # and losses, metrics, callbacks, and backend can still be used from Keras directly.
from keras.metrics import *
from keras import metrics
from sklearn.metrics import *
from keras import backend as K
from keras.backend import *
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from numpy.linalg import inv
from scipy.linalg import fractional_matrix_power
from keras.utils.np_utils import *
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import HierarchicalHAR_PBD


#Segmentation(Sliding Window)
def segmentation(mat,overlap_ratio,window_len):
    x_mocap_segment = np.zeros((1,66))
    x_emg_segment = np.zeros((1,4))
    y_segment = np.array([[0,0]])
    m = len(mat)    #number of rows
    n = len(mat[0])    #number of columns
    step = int(window_len*(1-overlap_ratio))
   
    
    length = 0    #length of the data
    num_frames = 0    #number of frames
    #if remain length of matrix is samller than a window length, the last window padding with 0
    while (m-length)>window_len:
        x_mocap_segment = np.concatenate((x_mocap_segment,mat[num_frames*step:num_frames*step+window_len,:66]),0)
        x_emg_segment = np.concatenate((x_emg_segment,mat[num_frames*step:num_frames*step+window_len,66:70]),0)
        y_HAR_segment = int(Counter(mat[num_frames*step:num_frames*step+window_len,70]).most_common(1)[0][0])
        #standing, sitting, walking and others all considered as transition
        #number of classes of PBD is 6, including 5 activities and 1 transition
        if y_HAR_segment==6 or y_HAR_segment==7 or y_HAR_segment==8:
            y_HAR_segment = 0 
        y_PDB_segment = int(Counter(mat[num_frames*step:num_frames*step+window_len,72]).most_common(1)[0][0])
        
        y_segment = np.concatenate((y_segment,[[y_HAR_segment,y_PDB_segment]]),0)
        num_frames += 1
        length = 90*(num_frames+1)  
    
    #padding last window with 0
    len_remain = m - length
    mat_padding = np.zeros([window_len - len_remain,n])
    mat_last = np.concatenate((mat[length:,:],mat_padding[:,:]),0)
    
    #final concatenation
    x_mocap_segment = np.concatenate((x_mocap_segment,mat_last[:,:66]),0)
    x_emg_segment = np.concatenate((x_emg_segment,mat_last[:,66:70]),0)
    y_HAR_segment = int(Counter(mat_last[:,70]).most_common(1)[0][0])
    if y_HAR_segment==6 or y_HAR_segment==7 or y_HAR_segment==8:
            y_HAR_segment = 0 
    y_PDB_segment = int(Counter(mat_last[:,72]).most_common(1)[0][0])
    y_segment = np.concatenate((y_segment,[[y_HAR_segment,y_PDB_segment]]),0)
    
    #reshape the data
    num_frames = num_frames + 1
    x_mocap_segment = np.reshape(x_mocap_segment[1:],(num_frames,window_len,66))
    x_emg_segment = np.reshape(x_emg_segment[1:],(num_frames,window_len,4))
    y_segment = y_segment[1:]

    return x_mocap_segment, x_emg_segment, y_segment


#load the data
def train_data_load(path, valid_patient):
    files_dir = os.listdir(path)    
    x_mocap = np.zeros((1,180,66))  
    x_emg = np.zeros((1,180,4))
    y = np.array([[0,0]])
    for file_dir in files_dir:
        if os.path.splitext(file_dir)[1] == '.mat':
            if os.path.splitext(file_dir)[0] != str(valid_patient):
                Data=hd.loadmat('/home/jojo/cen/Data/'+ file_dir)
                Matrix = Data['data']
                x_mocap_segment, x_emg_segment, y_segment = segmentation(Matrix,0.5,180)
                x_mocap = np.concatenate((x_mocap,x_mocap_segment),0)
                x_emg = np.concatenate((x_emg,x_emg_segment),0)
                y = np.concatenate((y,y_segment),0)
            
    x_mocap = x_mocap[1:]
    x_emg = x_emg[1:]*100
    y = y[1:]
    return x_mocap,x_emg,y

def val_data_load(path, valid_patient):
    files_dir = os.listdir(path)    
    x_mocap = np.zeros((1,180,66))  
    x_emg = np.zeros((1,180,4))
    y = np.array([[0,0]])
    for file_dir in files_dir:
        if os.path.splitext(file_dir)[1] == '.mat':
            if os.path.splitext(file_dir)[0] == str(valid_patient):
                Data=hd.loadmat('/home/jojo/cen/Data/'+ file_dir)
                Matrix = Data['data']
                x_mocap_segment, x_emg_segment, y_segment = segmentation(Matrix,0.5,180)
                x_mocap = np.concatenate((x_mocap,x_mocap_segment),0)
                x_emg = np.concatenate((x_emg,x_emg_segment),0)
                y = np.concatenate((y,y_segment),0)
            
    x_mocap = x_mocap[1:]
    x_emg = x_emg[1:]*100
    y = y[1:]
    return x_mocap,x_emg,y


#Data Augementation

#jittering(Gaussian Noise)
def gauss_noise(data, dev):
    # data(N*180*66)
    #dev is deviation
    noise = np.random.normal(0, dev, data.shape)
    data_jitternig = data + noise
    return data_jitternig

#Cropping
def cropping(data,prob,node_num):
    #prob is the probability of dropping
    time_step = 180
    num_samples = data.shape[0]

    for i in range(num_samples):
        drop_samples = np.random.randint(0, time_step, int(round(time_step*prob)))
        for j in drop_samples:
            drop_nodes = np.random.randint(0, node_num, int(round(node_num*prob)))
            for k in drop_nodes:
                data[i,j,k] = 0
                data[i,j,k+node_num] = 0
                data[i,j,k+node_num*2] = 0
    return data     


#LOSO(Leave-One-Subject-Out cross validation)

model = HierarchicalHAR_PBD.build_model(180,22,4,3,1,26,8,5,24,8,24,16,6,5,24,8,24,6,2)
for valid_patient in range(13,31):
    
    #Data Generation
    #original training data
    X_mocap_train, X_emg_train, y_train = train_data_load('/home/jojo/cen/Data',valid_patient)
    #original validation data
    X_mocap_valid, X_emg_valid, y_valid = val_data_load('/home/jojo/cen/Data',valid_patient)
    #data augmentation with Gaussian Noise
    X_mocap_train_g1 = gauss_noise(X_mocap_train, 0.05)
    X_emg_train_g1 = gauss_noise(X_emg_train, 0.05)
    y_train_g1 = y_train
    X_mocap_train_g2 = gauss_noise(X_mocap_train, 0.1)
    X_emg_train_g2 = gauss_noise(X_emg_train, 0.1)
    y_train_g2 = y_train
    #data augmentation with cropping
    X_mocap_train_c1 = cropping(X_mocap_train, 0.05, 22)
    X_emg_train_c1 = cropping(X_emg_train, 0.05, 4)
    y_train_c1 = y_train
    X_mocap_train_c2 = cropping(X_mocap_train, 0.1, 22)
    X_emg_train_c2 = cropping(X_emg_train, 0.1, 4)
    y_train_c2 = y_train
    #concatenate original data and augmentated data
    X_MoCap_train = np.concatenate((X_mocap_train, X_mocap_train_g1, X_mocap_train_g2, X_mocap_train_c1, X_mocap_train_c2), 0)
    X_EMG_train = np.concatenate((X_emg_train, X_emg_train_g1, X_emg_train_g2, X_emg_train_c1, X_emg_train_c2), 0)
    y_train = np.concatenate((y_train, y_train_g1, y_train_g2, y_train_c1, y_train_c2), 0)
    
    #print(X_MoCap_train.shape)
    #print(X_EMG_train.shape)
    
    
    #Input Transformation
    time_step = 180
    node_num = 22
    muscle_num = 4
    feature_num = 3
    AdjNorm_MoCap = utils.MakeGraph_MoCap()
    AdjNorm_EMG = utils.MakeGraph_EMG()
    mocap_graphtrain = utils.combine_mocap(AdjNorm_MoCap, X_MoCap_train, time_step, node_num, feature_num)
    emg_graphtrain = utils.combine_emg(AdjNorm_EMG, X_EMG_train, time_step, muscle_num, 1)    #only one number for each node
    mocap_graphvalid = utils.combine_mocap(AdjNorm_MoCap, X_mocap_valid, time_step, node_num, feature_num)
    emg_graphvalid = utils.combine_emg(AdjNorm_EMG, X_emg_valid, time_step, muscle_num, 1)
    

    #One Hot
    y_train_HAR = to_categorical(y_train[:,0])
    y_train_PBD = to_categorical(y_train[:,1])
    y_valid_HAR = to_categorical(y_valid[:,0])
    y_valid_PBD = to_categorical(y_valid[:,1])
    
    
    #Model Compile
    model.compile(optimizer=Adam(lr=5e-4,decay=1e-5),
              loss={'PBDout':utils.focal_loss(weights = utils.class_balance_weights(0.9999,
                                     [np.sum(y_train_PBD[:, 0]), np.sum(y_train_PBD[:, 1])]),
                                     gamma=2, num_class=2),
                    'HARout':'categorical_crossentropy'
                   },
              loss_weights={'PBDout': 1., 'HARout': 1.},
              metrics=['categorical_accuracy'])
    
    
    #Model Training
    model.fit({'mocapinputs':mocap_graphtrain,'emginputs':emg_graphtrain},
              {'PBDout': y_train_PBD, 'HARout': y_train_HAR},
              batch_size=150,
              epochs=100,
              callbacks=utils.build_callbacks('Model', str(valid_patient)),
              validation_data=((mocap_graphvalid,emg_graphvalid), (y_valid_HAR,y_valid_PBD))
             )
    
    #eliminate memory
    del X_mocap_train
    del X_emg_train
    del y_train
    del X_mocap_valid
    del X_emg_valid
    del y_valid
    del X_mocap_train_g1
    del X_emg_train_g1
    del y_train_g1
    del X_mocap_train_g2
    del X_emg_train_g2
    del y_train_g2
    del X_mocap_train_c1
    del X_emg_train_c1
    del y_train_c1
    del X_mocap_train_c2
    del X_emg_train_c2
    del y_train_c2
    del X_MoCap_train
    del X_EMG_train
    del mocap_graphtrain
    del emg_graphtrain
    del mocap_graphvalid
    del emg_graphvalid
    del y_train_HAR
    del y_train_PBD
    del y_valid_HAR
    del y_valid_PBD
    
    
