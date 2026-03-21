
import itertools
import subprocess
import pickle


import warnings
import time

import pandas as pd
import numpy as np

import pickle

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go

import lifelines
from lifelines.utils import concordance_index

import os

import tensorflow as tf
import tensorflow_probability as tfp

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

from tensorflow import keras
from tensorflow.keras import optimizers, initializers, regularizers, layers

from scipy.stats import norm, t, probplot, pearsonr
from scipy.special import gamma

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

# import thetaflow as thf
import modelnn2 as thf


# --- GPU Configuration ---
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def build_cox_nnet(dropout_rate = 0.1, ridge_penalty = 1.0e-2):
    cox_nnet_parameters = {
        "theta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": 0.0},
    }
    
    def partial_loglikelihood_loss(model, nn_output, data):
        # Unpack your data tuple
        X, y, delta = data
        theta = model.get_variable("theta", nn_output)
        
        # Shapes for broascasting
        y_col = tf.reshape(y, [-1, 1])
        y_row = tf.reshape(y, [1, -1])
        theta_col = tf.reshape(theta, [-1, 1])
        delta_col = tf.reshape(delta, [-1, 1])

        # Matrix form to replace the sum loop in the partial log-likelihoood
        R_matrix = tf.cast(y_row >= y_col, dtype = tf.float32)
        exp_theta = tf.math.exp(theta_col)
        risk_sum = tf.matmul(R_matrix, exp_theta)
        log_risk_sum = tf.math.log(risk_sum + 1e-7)
        
        pll = tf.reduce_sum( delta_col * (theta_col - log_risk_sum) )
        return -pll

    def neural_network(model, seed = None):
        initializer = initializers.GlorotNormal(seed = seed)
        ridge = tf.keras.regularizers.L2(l2 = ridge_penalty)
        
        # 1. The Hidden Layer (Feature Extractor)
        model.dense1 = layers.Dense(
            units = 132,
            activation = "tanh",     
            kernel_initializer = initializer,
            kernel_regularizer = ridge,
            use_bias = True,
            dtype = tf.float32, 
            name = "latent_representation"
        )
        
        # 2. The Dropout Layer
        model.dropout = layers.Dropout(rate = dropout_rate, seed = seed)
        
        # 3. The Cox Beta Layer (The brilliant realization)
        model.output_layer = layers.Dense(
            units = 1, 
            activation = None,
            use_bias = False,
            kernel_initializer = initializer,
            kernel_regularizer = ridge,
            dtype = tf.float32,
            name = "beta_coefficients"
        )
    
    def neural_network_call(model, x_input, training = False):
        x = model.dense1(x_input)
        x = model.dropout(x, training = training)
        return model.output_layer(x) # Returns the final theta scalar for each patient
    
    def neural_network_call_nolast(model, x_input):
        x = model.dense1(x_input)
        return x
    
    return cox_nnet_parameters, partial_loglikelihood_loss, neural_network, neural_network_call, neural_network_call_nolast

def train_eval_cox_nnet(X_train, data_train, cox_nnet_parameters, 
                        cox_nnet_partial_loglikelihood_loss, cox_nnet_neural_network, cox_nnet_network_call, cox_nnet_network_call_nolast,
                        epochs = 10000, shuffle = True, metrics_update_freq = 50,
                        fine_tune = False,
                        get_covariances = False,
                        validation = False, val_prop = 0.2,
                        optimizer_nn = optimizers.SGD(learning_rate = 0.1, momentum = 0.9, nesterov = True, clipnorm = 1.0),
                        train_batch_size = None, val_batch_size = None,
                        buffer_size = 4096, gradient_accumulation_steps = None,
                        early_stopping = True, early_stopping_tolerance = 1.0e-4, early_stopping_warmup = 1000,
                        reduce_lr = True, reduce_lr_warmup = 10,
                        reduce_lr_factor = 0.9, reduce_lr_min_delta = 1.0e-4, reduce_lr_patience = 5,
                        reduce_lr_cooldown = 20, reduce_lr_min_lr = 1e-5,
                        deterministic = True,
                        verbose = 1, print_freq = 100):
    """
        Get data and configuration and trains a completely new cox-nnet model and returns it.
    """
    cox_nnet_model = thf.ModelNN(cox_nnet_parameters, cox_nnet_partial_loglikelihood_loss,
                                 cox_nnet_neural_network, cox_nnet_network_call,
                                 cox_nnet_network_call_nolast, input_dim = (None, X_train.shape[1]), seed = 10)
    cox_nnet_model.pre_train_model(X_train, data_train,
                                   epochs = 1000, shuffle = True,
                                   optimizer_nn = optimizers.Adam(learning_rate = 0.1),
                                   verbose = 1, track_time = True)
    cox_nnet_model.train_model(X_train, data_train,
                               epochs = epochs, shuffle = shuffle, metrics_update_freq = metrics_update_freq,
                               fine_tune = fine_tune,
                               get_covariances = get_covariances,
                               validation = validation, val_prop = val_prop,
                               optimizer_nn = optimizer_nn,
                               train_batch_size = train_batch_size, val_batch_size = val_batch_size,
                               buffer_size = buffer_size, gradient_accumulation_steps = gradient_accumulation_steps,
                               early_stopping = early_stopping, early_stopping_tolerance = early_stopping_tolerance, early_stopping_warmup = early_stopping_warmup,
                               reduce_lr = reduce_lr, reduce_lr_warmup = reduce_lr_warmup,
                               reduce_lr_factor = reduce_lr_factor, reduce_lr_min_delta = reduce_lr_min_delta, reduce_lr_patience = reduce_lr_patience,
                               reduce_lr_cooldown = reduce_lr_cooldown, reduce_lr_min_lr = reduce_lr_min_lr,
                               deterministic = deterministic,
                               verbose = verbose, print_freq = print_freq)
    return cox_nnet_model

def main():
    # 1. Load the payload
    with open("grid_search_payload.pkl", "rb") as f:
        payload = pickle.load(f)
        
    X = payload["X"]
    y = payload["y"]
    delta = payload["delta"]
    params = payload["params"]
    n_splits = payload["n_splits"]
    epochs = payload["epochs"]

    # Extract the specific parameters for this run
    dropout_rate = params.get('dropout_rate', 0.1)
    ridge_penalty = params.get('ridge_penalty', 1e-2)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_c_indices = []
    
    # 2. Run the K-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Data Splitting
        X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
        y_train, y_val = y[train_idx], y[val_idx]
        delta_train, delta_val = delta[train_idx], delta[val_idx]
        
        X_train = tf.cast(X_train, dtype=tf.float32)
        X_val = tf.cast(X_val, dtype=tf.float32)
        
        data_train = [
            tf.constant(y_train, shape = (len(y_train), 1), dtype=tf.float32), 
            tf.constant(delta_train, shape = (len(delta_train), 1), dtype=tf.float32)
        ]
        
        # Rebuild Architecture
        cox_nnet_params, pll_loss, nn_arch, nn_call, nn_call_nolast = build_cox_nnet(
            dropout_rate=dropout_rate, 
            ridge_penalty=ridge_penalty
        )
        
        # Optimizer must be created fresh for each fold
        local_optimizer = optimizers.SGD(learning_rate = 0.1, momentum = 0.9, nesterov = True, clipnorm = 1.0)
        
        # Train the Cox-nnet model
        model = train_eval_cox_nnet(
            X_train = X_train,
            data_train = data_train,
            cox_nnet_parameters = cox_nnet_params,
            cox_nnet_partial_loglikelihood_loss = pll_loss,
            cox_nnet_neural_network = nn_arch,
            cox_nnet_network_call = nn_call,
            cox_nnet_network_call_nolast = nn_call_nolast,
            optimizer_nn = local_optimizer,
            epochs = epochs,
            verbose = True, # Keep console clean so it runs silently in the background
            print_freq = 100
        )
        
        # Evaluate C-Index
        theta_val = model.predict(X_val)["theta"].numpy().flatten()
        fold_c_index = concordance_index(
            event_times=y_val.flatten(),
            predicted_scores=-theta_val,
            event_observed=delta_val.flatten()
        )
        
        fold_c_indices.append(fold_c_index)
        print(f"Fold {fold + 1} C-index: {fold_c_index:.4f}")
        
    # 3. Calculate average and save the result for the main script
    avg_c_index = np.mean(fold_c_indices)
    
    result_data = {
        "avg_c_index": avg_c_index,
        "fold_c_indices": fold_c_indices
    }
    
    with open("grid_search_results.pkl", "wb") as f:
        pickle.dump(result_data, f)

if __name__ == "__main__":
    main()