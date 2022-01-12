# Code Author: Guanting Cen, Chongyang Wang (My PhD was supported by UCL Overseas Research Scholarship and Graduate Research Scholarship, and partially by the EU Future and Emerging Technologies (FET) Proactive Programme H2020-EU.1.2.2 (Grant agreement 824160; EnTimeMent).
# The code is created on 01/12/2021

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # enable a clean command line  window. The thing is to define os. before importing tensorflow.
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
import tensorflow as tf
import scipy.io
import h5py
import keras
#import xlwt as xw
import hdf5storage
from sklearn.metrics import *
from keras.layers import *
from keras.layers.core import *
from keras.models import *
from keras.regularizers import *
from keras.optimizers import *
from keras.losses import *
from keras.metrics import *
from keras import backend as K
from keras.callbacks import EarlyStopping
from scipy.linalg import fractional_matrix_power

from tensorflow.compat.v1.keras.layers import CuDNNLSTM as LSTM
import utils

# the following three functions are needed for the hierarchical connection between HAR and PBD modules, and for the Lambda layer to perform the graph normalization per GCN layer.

def HARExtend(nodefeature):
    # Extend the HAR output from (feature_dim,) to (time_step,body_num,feature_num,), in order to be concatenated with the raw input for the PBD module.
    # Define the value for time_step,body_num,class_num
    time_step = 180
    body_num = 22
    class_num = 6
    
    HARExtend = K.argmax(nodefeature,-1)
    HARExtend = K.one_hot(HARExtend,class_num)
    HARextend = K.expand_dims(HARExtend, axis=1)
    HARextend = K.expand_dims(HARextend, axis=1)
    HARextend = K.tile(HARextend, n=[1,time_step, body_num, 1, ])
    
    return HARextend


def output_of_adjmul(input_shape):
    return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])

def adjmul(x):
    AdjNorm = utils.MakeGraph() # refer to utils.py for this function.
    x = tf.cast(x, tf.float64) # this step could be removed in earlier Tensorflow versions.
    return tf.matmul(AdjNorm,x)

def build_model(timestep,body_num,feature_dim,gcn_units_HAR,lstm_units_HAR,gcn_units_PBD,lstm_units_PBD,num_class_HAR,num_class_PBD):
    # timestep=the length of current input data segment.
    # body_num=the number of nodes/joints of the input graph.
    # feature_num=the feature dimension of each node/joint.
    # gcn/lstm_units=the units of gcn and lstm layers.
    # num_class=the number of categories.

    # Mutual Input
    inputs = Input(shape=(timestep, body_num, feature_dim,), name='maininputs')

    # HAR LSTM:
    HARsingleinput = Input(shape=(timestep, gcn_units_HAR * body_num))
    HARLSTM1 = LSTM(lstm_units_HAR, return_sequences=True, name='HARLSTM1')(HARsingleinput) # refer to gc-LSTM for how to use GPU for LSTM  using Tensorflow>=2.0
    HARDropout1 = Dropout(0.5)(HARLSTM1)
    HARLSTM2 = LSTM(lstm_units_HAR, return_sequences=True, name='HARLSTM2')(HARDropout1)
    HARDropout2 = Dropout(0.5)(HARLSTM2)
    HARLSTM3 = LSTM(lstm_units_HAR, return_sequences=False, name='HARLSTM3')(HARDropout2)
    HARDropout3 = Dropout(0.5)(HARLSTM3)
    HARLSTM = Model(inputs=[HARsingleinput], outputs=[HARDropout3])

    # HAR GCN
    HARDense1 = TimeDistributed(Conv1D(gcn_units_HAR, 1, activation='relu'), name='HARGCN1')(inputs)
    HARDense1 = Dropout(0.5)(HARDense1)
    HARDense2 = Reshape((timestep, gcn_units_HAR * body_num), )(HARDense1)
    
    HARTemporaloutput = HARLSTM(HARDense2)
    HARTemporaloutput1 = Dense(num_class_HAR, activation='softmax', name='HARout')(HARTemporaloutput)
    
#     print(HARTemporaloutput1.shape)
    
    # PBD LSTM:
    PBDsingleinput = Input(shape=(timestep, body_num * gcn_units_PBD))
    PBDLSTM1 = LSTM(lstm_units_PBD, return_sequences=True)(PBDsingleinput)
    PBDDropout1 = Dropout(0.5)(PBDLSTM1)
    PBDLSTM2 = LSTM(lstm_units_PBD, return_sequences=True)(PBDDropout1)
    PBDDropout2 = Dropout(0.5)(PBDLSTM2)
    PBDLSTM3 = LSTM(lstm_units_PBD, return_sequences=False)(PBDDropout2)
    PBDDropout3 = Dropout(0.5)(PBDLSTM3)
    PBDLSTM = Model(inputs=[PBDsingleinput], outputs=[PBDDropout3])
    
    
    # PBD GCN
    HARextend = Lambda(HARExtend,output_shape=(timestep,body_num,num_class_HAR))(HARTemporaloutput1)
#     print(HARextend.shape)
    PBDinputs = concatenate([inputs, HARextend], axis=-1)
#     print(PBDinputs.shape)
    PBDDense1 = TimeDistributed(Conv1D(gcn_units_PBD, 1, activation='relu'))(PBDinputs)
    PBDDense1 = Dropout(0.5)(PBDDense1)
#     print(PBDDense1.shape)
    PBDDense2 = Lambda(adjmul, output_shape=output_of_adjmul)(PBDDense1)
    PBDDense2 = TimeDistributed(Conv1D(gcn_units_PBD, 1, activation='relu'))(PBDDense2)
    PBDDense2 = Dropout(0.5)(PBDDense2)
    PBDDense3 = Lambda(adjmul, output_shape=output_of_adjmul)(PBDDense2)
    PBDDense3 = TimeDistributed(Conv1D(gcn_units_PBD, 1, activation='relu'))(PBDDense3)
    PBDDense3 = Dropout(0.5)(PBDDense3)
    PBDDense4 = Reshape((timestep, body_num * gcn_units_PBD), )(PBDDense3)
    
    PBDTemporaloutput = PBDLSTM(PBDDense4)
    PBDTemporaloutput1 = Dense(num_class_PBD, activation='softmax', name='PBDout')(PBDTemporaloutput)
    
    
    HARmodel = Model(inputs=[inputs], outputs=[HARTemporaloutput1])    #HAR model, output the label of activity classes
    #model = Model(inputs=[inputs], outputs=[PBDTemporaloutput1])
    model = Model(inputs=[inputs], outputs=[HARTemporaloutput1, PBDTemporaloutput1])    #final model, one input and two output(HAR,PDB)
    
    print(HARTemporaloutput1.shape)
    print(PBDTemporaloutput1.shape)

    return model, HARmodel