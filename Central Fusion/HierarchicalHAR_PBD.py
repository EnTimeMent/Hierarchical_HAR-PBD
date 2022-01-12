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

def build_model(timestep, body_num, muscle_num, feature_dim_mocap, feature_dim_emg, gcn_units_HAR_mocap,gcn_units_HAR_emg, 
                units_HAR_central, lstm_units_HAR_mocap, lstm_units_HAR_emg, lstm_units_HAR_central, gcn_units_PBD_mocap,gcn_units_PBD_emg,
                units_PBD_central, lstm_units_PBD_mocap, lstm_units_PBD_emg, lstm_units_PBD_central, num_class_HAR,num_class_PBD):
    # timestep=the length of current input data segment.
    # body_num/muscle_num=the number of nodes/joints of the input graph.
    # feature_dim_mocap=the feature dimension of each node/joint of MoCap graph.
    # feature_dim_emg=the feature dimension of each node/joint of EMG graph.
    # units_HAR_central=the dimension of projected features 
    # gcn/lstm_units=the units of gcn and lstm layers.
    # num_class=the number of categories.

    # Mutual Input
    inputs_mocap = Input(shape=(timestep, body_num, feature_dim_mocap,), name='mocapinputs')
    inputs_emg = Input(shape=(timestep, muscle_num, feature_dim_emg,), name='emginputs')
    
    
    # ================ HAR ================
    
    # == MoCap Graph HAR ==
    # HAR GCN
    HARDense1_MoCap = TimeDistributed(Conv1D(gcn_units_HAR_mocap, 1, activation='relu'), name='HARGCN1_MoCap')(inputs_mocap)
    HARDense1_MoCap = Dropout(0.5)(HARDense1_MoCap)
    HARDense2_MoCap = Reshape((timestep, gcn_units_HAR_mocap * body_num), )(HARDense1_MoCap)
       
    
    
    # == EMG Graph HAR==
    # HAR GCN
    HARDense1_EMG = TimeDistributed(Conv1D(gcn_units_HAR_emg, 1, activation='relu'), name='HARGCN1_EMG')(inputs_emg)
    HARDense1_EMG = Dropout(0.5)(HARDense1_EMG)
    HARDense2_EMG = Reshape((timestep, gcn_units_HAR_emg * muscle_num), )(HARDense1_EMG)
    

    
    # == Central Fusion HAR ==
    # HAR LSTM
    #input the central layer representation to LSTM model
    HARsingleinput_Central = Input(shape=(timestep, units_HAR_central * (body_num+muscle_num)))
    HARLSTM1_Central = LSTM(lstm_units_HAR_central, return_sequences=True, name='HARLSTM1_Central')(HARsingleinput_Central) 
    HARDropout1_Central = Dropout(0.5)(HARLSTM1_Central)
    HARLSTM2_Central = LSTM(lstm_units_HAR_central, return_sequences=True, name='HARLSTM2_Central')(HARDropout1_Central)
    HARDropout2_Central = Dropout(0.5)(HARLSTM2_Central)
    HARLSTM3_Central = LSTM(lstm_units_HAR_central, return_sequences=False, name='HARLSTM3_Central')(HARDropout2_Central)
    HARDropout3_Central = Dropout(0.5)(HARLSTM3_Central)
    HARLSTM_Central = Model(inputs=[HARsingleinput_Central], outputs=[HARDropout3_Central])
    
    
    # HAR Central Connection
    #mapping last dimension into the same for each graph input and concatenate together; connect the hidden central layers with sum  
    
    #central layer1
    HARinputs_mocap_Central = TimeDistributed(Conv1D(units_HAR_central, 1, activation='relu'))(inputs_mocap)
    HARinputs_emg_Central = TimeDistributed(Conv1D(units_HAR_central, 1, activation='relu'))(inputs_emg)
    HARCentral1 = tf.concat((HARinputs_mocap_Central,HARinputs_emg_Central),axis=2)
    HARDense1_Central = Dense(units_HAR_central, activation='softmax')(HARCentral1)
    #central layer2
    HARDense1_MoCap_Central = TimeDistributed(Conv1D(units_HAR_central, 1, activation='relu'))(HARDense1_MoCap)
    HARDense1_EMG_Central = TimeDistributed(Conv1D(units_HAR_central, 1, activation='relu'))(HARDense1_EMG)
    HARCentral2 = tf.concat((HARDense1_MoCap_Central,HARDense1_EMG_Central),axis=2)
    HARDense2_Central = HARDense1_Central + HARCentral2
    
    #central layer3, reshape and input to LSTM
    HARDense3_Central = Reshape((timestep, units_HAR_central * (body_num+muscle_num)), )(HARDense2_Central)
    
    
    HARTemporaloutput_Central = HARLSTM_Central(HARDense3_Central)
    HARTemporaloutput1_Central = Dense(num_class_HAR, activation='softmax', name='HARout')(HARTemporaloutput_Central)

    
    
    
    # =========== PBD ===========
    
    # == MoCap Graph PBD ==
    # PBD GCN
    HARextend_MoCap = Lambda(HARExtend_MoCap,output_shape=(timestep,body_num,num_class_HAR))(HARTemporaloutput1_Central)
    
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
    

    
    # == EMG Graph PBD ==
    # PBD GCN
    HARextend_EMG = Lambda(HARExtend_EMG,output_shape=(timestep,muscle_num,num_class_HAR))(HARTemporaloutput1_Central)
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
    
    
    
    # ==Central Fusion PBD ==
    # PBD LSTM
    #input the central layer representation to LSTM model
    PBDsingleinput_Central = Input(shape=(timestep, units_HAR_central * (body_num+muscle_num)))
    PBDLSTM1_Central = LSTM(lstm_units_HAR_central, return_sequences=True, name='PBDLSTM1_Central')(PBDsingleinput_Central) 
    PBDDropout1_Central = Dropout(0.5)(PBDLSTM1_Central)
    PBDLSTM2_Central = LSTM(lstm_units_HAR_central, return_sequences=True, name='PBDLSTM2_Central')(PBDDropout1_Central)
    PBDDropout2_Central = Dropout(0.5)(PBDLSTM2_Central)
    PBDLSTM3_Central = LSTM(lstm_units_HAR_central, return_sequences=False, name='PBDLSTM3_Central')(PBDDropout2_Central)
    PBDDropout3_Central = Dropout(0.5)(PBDLSTM3_Central)
    PBDLSTM_Central = Model(inputs=[PBDsingleinput_Central], outputs=[PBDDropout3_Central])
    
    
    # PBD Central Connection
    #mapping last dimension into the same for each graph input and concatenate together; connect the hidden central layers with sum  
    #central layer1
    PBDinputs_mocap_Central = TimeDistributed(Conv1D(units_PBD_central, 1, activation='relu'))(PBDinputs_MoCap)
    PBDinputs_emg_Central = TimeDistributed(Conv1D(units_PBD_central, 1, activation='relu'))(PBDinputs_EMG)
    PBDCentral1 = tf.concat((PBDinputs_mocap_Central,PBDinputs_emg_Central),axis=2)
    PBDDense1_Central = Dense(units_PBD_central, activation='softmax')(PBDCentral1)
    #central layer2
    PBDDense1_MoCap_Central = TimeDistributed(Conv1D(units_PBD_central, 1, activation='relu'))(PBDDense1_MoCap)
    PBDDense1_EMG_Central = TimeDistributed(Conv1D(units_PBD_central, 1, activation='relu'))(PBDDense1_EMG)
    PBDCentral2 = tf.concat((PBDDense1_MoCap_Central,PBDDense1_EMG_Central),axis=2)
    PBDDense2_Central = PBDDense1_Central + PBDCentral2
    PBDDense2_Central = Dense(units_PBD_central, activation='softmax')(PBDDense2_Central)
    #central layer3
    PBDDense2_MoCap_Central = TimeDistributed(Conv1D(units_PBD_central, 1, activation='relu'))(PBDDense2_MoCap)
    PBDDense2_EMG_Central = TimeDistributed(Conv1D(units_PBD_central, 1, activation='relu'))(PBDDense2_EMG)
    PBDCentral3 = tf.concat((PBDDense2_MoCap_Central,PBDDense2_EMG_Central),axis=2)
    PBDDense3_Central = PBDDense2_Central + PBDCentral3
    PBDDense3_Central = Dense(units_PBD_central, activation='softmax')(PBDDense3_Central)
    #central layer4
    PBDDense3_MoCap_Central = TimeDistributed(Conv1D(units_PBD_central, 1, activation='relu'))(PBDDense3_MoCap)
    PBDDense3_EMG_Central = TimeDistributed(Conv1D(units_PBD_central, 1, activation='relu'))(PBDDense3_EMG)
    PBDCentral4 = tf.concat((PBDDense3_MoCap_Central,PBDDense3_EMG_Central),axis=2)
    PBDDense4_Central = PBDDense3_Central + PBDCentral4
    
    #resnet
    PBDDense4_Central = PBDDense4_Central+PBDDense1_Central
    
    #central layer5, reshape and input to LSTM
    PBDDense5_Central = Reshape((timestep, units_HAR_central * (body_num+muscle_num)), )(PBDDense4_Central)
    
    
    PBDTemporaloutput_Central = PBDLSTM_Central(PBDDense5_Central)
    PBDTemporaloutput1_Central = Dense(num_class_PBD, activation='softmax', name='PBDout')(PBDTemporaloutput_Central)
    print(HARTemporaloutput1_Central.shape)
    print(PBDTemporaloutput1_Central.shape)

    
    #HARmodel = Model(inputs=[inputs_mocap,inputs_emg], outputs=[HARTemporaloutput2])    #HAR model, output the label of activity classes
    #model = Model(inputs=[inputs], outputs=[PBDTemporaloutput1])
    model = Model(inputs=[inputs_mocap,inputs_emg], outputs=[HARTemporaloutput1_Central, PBDTemporaloutput1_Central])    #final model, one input and two output(HAR,PDB)

    return model