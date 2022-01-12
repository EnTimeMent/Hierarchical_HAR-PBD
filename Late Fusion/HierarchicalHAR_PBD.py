# Code Author: Guanting Cen, Chongyang Wang (My PhD was supported by UCL Overseas Research Scholarship and Graduate Research Scholarship, 
# and partially by the EU Future and Emerging Technologies (FET) Proactive Programme H2020-EU.1.2.2 (Grant agreement 824160; EnTimeMent).

# Revision Date: 26 12 2021


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

def HARExtend_MoCap(nodefeature):
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

def HARExtend_EMG(nodefeature):
    # Extend the HAR output from (feature_dim,) to (time_step,body_num,feature_num,), in order to be concatenated with the raw input for the PBD module.
    # Define the value for time_step,body_num,class_num
    time_step = 180
    muscle_num = 4
    class_num = 6
    
    HARExtend = K.argmax(nodefeature,-1)
    HARExtend = K.one_hot(HARExtend,class_num)
    HARextend = K.expand_dims(HARExtend, axis=1)
    HARextend = K.expand_dims(HARextend, axis=1)
    HARextend = K.tile(HARextend, n=[1,time_step, muscle_num, 1, ])
    
    return HARextend

def output_of_adjmul(input_shape):
    return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])

def adjmul_mocap(x):
    AdjNorm = utils.MakeGraph_MoCap() # refer to utils.py for this function.
    x = tf.cast(x, tf.float64) # this step could be removed in earlier Tensorflow versions.
    return tf.matmul(AdjNorm,x)

def adjmul_emg(x):
    AdjNorm = utils.MakeGraph_EMG() # refer to utils.py for this function.
    x = tf.cast(x, tf.float64) # this step could be removed in earlier Tensorflow versions.
    return tf.matmul(AdjNorm,x)

def build_model(timestep, body_num, muscle_num, feature_dim_mocap, feature_dim_emg, gcn_units_HAR_mocap, gcn_units_HAR_emg,
                lstm_units_HAR_mocap, lstm_units_HAR_emg, gcn_units_PBD_mocap,gcn_units_PBD_emg, lstm_units_PBD_mocap,
                lstm_units_PBD_emg, num_class_HAR,num_class_PBD):
    # timestep=the length of current input data segment.
    # body_num/muscle_num=the number of nodes/joints of the input graph.
    # feature_dim_mocap=the feature dimension of each node/joint of MoCap graph.
    # feature_dim_emg=the feature dimension of each node/joint of EMG graph.
    # gcn/lstm_units=the units of gcn and lstm layers.
    # num_class=the number of categories.

    # Mutual Input
    inputs_mocap = Input(shape=(timestep, body_num, feature_dim_mocap,), name='mocapinputs')
    inputs_emg = Input(shape=(timestep, muscle_num, feature_dim_emg,), name='emginputs')
    

    
    ######## MoCap Graph HAR ########
    # HAR LSTM:
    HARsingleinput_MoCap = Input(shape=(timestep, gcn_units_HAR_mocap * body_num))
    HARLSTM1_MoCap = LSTM(lstm_units_HAR_mocap, return_sequences=True, name='HARLSTM1_MoCap')(HARsingleinput_MoCap) # refer to gc-LSTM for how to use GPU for LSTM  using Tensorflow>=2.0
    HARDropout1_MoCap = Dropout(0.5)(HARLSTM1_MoCap)
    HARLSTM2_MoCap = LSTM(lstm_units_HAR_mocap, return_sequences=True, name='HARLSTM2_MoCap')(HARDropout1_MoCap)
    HARDropout2_MoCap = Dropout(0.5)(HARLSTM2_MoCap)
    HARLSTM3_MoCap = LSTM(lstm_units_HAR_mocap, return_sequences=False, name='HARLSTM3_MoCap')(HARDropout2_MoCap)
    HARDropout3_MoCap = Dropout(0.5)(HARLSTM3_MoCap)
    HARLSTM_MoCap = Model(inputs=[HARsingleinput_MoCap], outputs=[HARDropout3_MoCap])

    # HAR GCN
    HARDense1_MoCap = TimeDistributed(Conv1D(gcn_units_HAR_mocap, 1, activation='relu'), name='HARGCN1_MoCap')(inputs_mocap)
    HARDense1_MoCap = Dropout(0.5)(HARDense1_MoCap)
    HARDense2_MoCap = Reshape((timestep, gcn_units_HAR_mocap * body_num), )(HARDense1_MoCap)
    
    HARTemporaloutput_MoCap = HARLSTM_MoCap(HARDense2_MoCap)
    #print(HARTemporaloutput_MoCap.shape)
    #HARTemporaloutput1_MoCap = Dense(num_class_HAR, activation='softmax', name='HARout_MoCap')(HARTemporaloutput_MoCap)
    
    
    
    ######## EMG Graph HAR ########
    #HAR LSTM
    HARsingleinput_EMG = Input(shape=(timestep, gcn_units_HAR_emg * muscle_num))
    HARLSTM1_EMG = LSTM(lstm_units_HAR_emg, return_sequences=True, name='HARLSTM1_EMG')(HARsingleinput_EMG) # refer to gc-LSTM for how to use GPU for LSTM  using Tensorflow>=2.0
    HARDropout1_EMG = Dropout(0.5)(HARLSTM1_EMG)
    HARLSTM2_EMG = LSTM(lstm_units_HAR_emg, return_sequences=True, name='HARLSTM2_EMG')(HARDropout1_EMG)
    HARDropout2_EMG = Dropout(0.5)(HARLSTM2_EMG)
    HARLSTM3_EMG = LSTM(lstm_units_HAR_emg, return_sequences=False, name='HARLSTM3_EMG')(HARDropout2_EMG)
    HARDropout3_EMG = Dropout(0.5)(HARLSTM3_EMG)
    HARLSTM_EMG = Model(inputs=[HARsingleinput_EMG], outputs=[HARDropout3_EMG])

    # HAR GCN
    HARDense1_EMG = TimeDistributed(Conv1D(gcn_units_HAR_emg, 1, activation='relu'), name='HARGCN1_EMG')(inputs_emg)
    HARDense1_EMG = Dropout(0.5)(HARDense1_EMG)
    HARDense2_EMG = Reshape((timestep, gcn_units_HAR_emg * muscle_num), )(HARDense1_EMG)
    
    HARTemporaloutput_EMG = HARLSTM_EMG(HARDense2_EMG)
    #print(HARTemporaloutput_EMG.shape)
    #HARTemporaloutput1_EMG = Dense(num_class_HAR, activation='softmax', name='HARout_EMG')(HARTemporaloutput_EMG)

    
    
    ######## Late Fusion HAR ########
    HARTemporaloutput1_fusion = tf.concat([HARTemporaloutput_MoCap,HARTemporaloutput_EMG],axis=-1)    
    #print(HARTemporaloutput1_fusion.shape)
    HARTemporaloutput1 = Dense(num_class_HAR,activation='softmax', name='HARout')(HARTemporaloutput1_fusion)
    print(HARTemporaloutput1.shape)
    
    
    
    
    ######## MoCap Graph PBD ########
    # PBD LSTM:
    PBDsingleinput_MoCap = Input(shape=(timestep, body_num * gcn_units_PBD_mocap))
    PBDLSTM1_MoCap = LSTM(lstm_units_PBD_mocap, return_sequences=True)(PBDsingleinput_MoCap)
    PBDDropout1_MoCap = Dropout(0.5)(PBDLSTM1_MoCap)
    PBDLSTM2_MoCap = LSTM(lstm_units_PBD_mocap, return_sequences=True)(PBDDropout1_MoCap)
    PBDDropout2_MoCap = Dropout(0.5)(PBDLSTM2_MoCap)
    PBDLSTM3_MoCap = LSTM(lstm_units_PBD_mocap, return_sequences=False)(PBDDropout2_MoCap)
    PBDDropout3_MoCap = Dropout(0.5)(PBDLSTM3_MoCap)
    PBDLSTM_MoCap = Model(inputs=[PBDsingleinput_MoCap], outputs=[PBDDropout3_MoCap])
    
    
    # PBD GCN
    HARextend_MoCap = Lambda(HARExtend_MoCap,output_shape=(timestep,body_num,num_class_HAR))(HARTemporaloutput1)
    
    PBDinputs_MoCap = concatenate([inputs_mocap, HARextend_MoCap], axis=-1)
    PBDDense1_MoCap = TimeDistributed(Conv1D(gcn_units_PBD_mocap, 1, activation='relu'))(PBDinputs_MoCap)
    PBDDense1_MoCap = Dropout(0.5)(PBDDense1_MoCap)
    PBDDense2_MoCap = Lambda(adjmul_mocap, output_shape=output_of_adjmul)(PBDDense1_MoCap)
    PBDDense2_MoCap = TimeDistributed(Conv1D(gcn_units_PBD_mocap, 1, activation='relu'))(PBDDense2_MoCap)
    PBDDense2_MoCap = Dropout(0.5)(PBDDense2_MoCap)
    PBDDense3_MoCap = Lambda(adjmul_mocap, output_shape=output_of_adjmul)(PBDDense2_MoCap)
    PBDDense3_MoCap = TimeDistributed(Conv1D(gcn_units_PBD_mocap, 1, activation='relu'))(PBDDense3_MoCap)
    PBDDense3_MoCap = Dropout(0.5)(PBDDense3_MoCap)
    PBDDense4_MoCap = Reshape((timestep, body_num * gcn_units_PBD_mocap), )(PBDDense3_MoCap)
    
    PBDTemporaloutput_MoCap = PBDLSTM_MoCap(PBDDense4_MoCap)
    #PBDTemporaloutput1_MoCap = Dense(num_class_PBD, activation='softmax', name='PBDout_MoCap')(PBDTemporaloutput_MoCap)
    
    
    ######## EMG Graph PBD ########
    # PBD LSTM:
    PBDsingleinput_EMG = Input(shape=(timestep, muscle_num * gcn_units_PBD_emg))
    PBDLSTM1_EMG = LSTM(lstm_units_PBD_emg, return_sequences=True)(PBDsingleinput_EMG)
    PBDDropout1_EMG = Dropout(0.5)(PBDLSTM1_EMG)
    PBDLSTM2_EMG = LSTM(lstm_units_PBD_emg, return_sequences=True)(PBDDropout1_EMG)
    PBDDropout2_EMG = Dropout(0.5)(PBDLSTM2_EMG)
    PBDLSTM3_EMG = LSTM(lstm_units_PBD_emg, return_sequences=False)(PBDDropout2_EMG)
    PBDDropout3_EMG = Dropout(0.5)(PBDLSTM3_EMG)
    PBDLSTM_EMG = Model(inputs=[PBDsingleinput_EMG], outputs=[PBDDropout3_EMG])
    
    
    # PBD GCN
    HARextend_EMG = Lambda(HARExtend_EMG,output_shape=(timestep,muscle_num,num_class_HAR))(HARTemporaloutput1)
    PBDinputs_EMG = concatenate([inputs_emg, HARextend_EMG], axis=-1)
    PBDDense1_EMG = TimeDistributed(Conv1D(gcn_units_PBD_emg, 1, activation='relu'))(PBDinputs_EMG)
    PBDDense1_EMG = Dropout(0.5)(PBDDense1_EMG)
    PBDDense2_EMG = Lambda(adjmul_emg, output_shape=output_of_adjmul)(PBDDense1_EMG)
    PBDDense2_EMG = TimeDistributed(Conv1D(gcn_units_PBD_emg, 1, activation='relu'))(PBDDense2_EMG)
    PBDDense2_EMG = Dropout(0.5)(PBDDense2_EMG)
    PBDDense3_EMG = Lambda(adjmul_emg, output_shape=output_of_adjmul)(PBDDense2_EMG)
    PBDDense3_EMG = TimeDistributed(Conv1D(gcn_units_PBD_emg, 1, activation='relu'))(PBDDense3_EMG)
    PBDDense3_EMG = Dropout(0.5)(PBDDense3_EMG)
    PBDDense4_EMG = Reshape((timestep, muscle_num * gcn_units_PBD_emg), )(PBDDense3_EMG)
    
    PBDTemporaloutput_EMG = PBDLSTM_EMG(PBDDense4_EMG)
    #PBDTemporaloutput1_EMG = Dense(num_class_PBD, activation='softmax', name='PBDout_EMG')(PBDTemporaloutput_EMG)
    
    
    ######## Late Fusion PBD ########
    PBDTemporaloutput1_fusion = tf.concat([PBDTemporaloutput_MoCap,PBDTemporaloutput_EMG],axis=-1)  
    #print(PBDTemporaloutput1_fusion.shape)
    PBDTemporaloutput1 = Dense(num_class_PBD,activation='softmax', name='PBDout')(PBDTemporaloutput1_fusion)
    print(PBDTemporaloutput1.shape)
    
    #HARmodel = Model(inputs=[inputs_mocap,inputs_emg], outputs=[HARTemporaloutput2])    #HAR model, output the label of activity classes
    #model = Model(inputs=[inputs], outputs=[PBDTemporaloutput1])
    model = Model(inputs=[inputs_mocap,inputs_emg], outputs=[HARTemporaloutput1, PBDTemporaloutput1])    #final model, one input and two output(HAR,PDB)

    return model
