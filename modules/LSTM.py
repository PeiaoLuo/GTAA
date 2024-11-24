from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import os

#formulate the structure of data to fit the requirement of LSTM model input
def get_dataset(df: pd.DataFrame, back: int):
    X, Y = [], []
    for i in range(len(df)-back-1):
        a = (np.array(df.iloc[i:i+back,:]))
        X.append(a)
        Y.append(df.iloc[i+back,0])
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(df.iloc[len(df)-back:,:])
    return X, Y, Z

def train(df: pd.DataFrame, test_length: int, back: int, settings: list,):
    
    batch_size = settings[0]
    epochs = settings[1]
    
    processed_df = df.copy()
    #preprocess data
    # scaler_1 = RobustScaler()
    # scaler_1 = MinMaxScaler()
    # processed_df = df.copy()
    # processed_df.iloc[:,1:] = scaler_1.fit_transform(df.iloc[:,1:])
    
    # scaler_2 = RobustScaler()
    # scaler_2 = MinMaxScaler()
    # robust_y = np.reshape(np.array(df.iloc[:,0]), (-1,1))
    # robust_y = scaler_2.fit_transform(robust_y)
    # processed_df.iloc[:,0] = robust_y
    #train test split
    X, Y, Z = get_dataset(df=processed_df, back=back)
    Z = np.reshape(Z, (1,Z.shape[0],Z.shape[1]))
    if test_length>0:
        train_X = X[:-test_length]
        train_Y = Y[:-test_length]
        test_X = X[-test_length:]
        test_Y = Y[-test_length:]
    else:
        train_X = X
        train_Y = Y
        
    # train_X = np.reshape(train_X, (train_X.shape[0], back, train_X.shape[1]))
    # test_X = np.reshape(test_X, (test_X.shape[0], back, test_X.shape[1]))

    #model implementation
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=2)
    
    train_pred = model.predict(train_X)
    if test_length > 0:
        test_pred = model.predict(test_X)
    else:
        test_pred = None
    Z = model.predict(Z)
    #MSE calc
    Y_train_hat = train_pred
    # Y_train_hat = scaler_2.inverse_transform(Y_train_hat)
    
    Y_train_real = np.array(df.iloc[back+1:-test_length,0])
    Y_train_real = np.reshape(Y_train_real, Y_train_hat.shape)
    if test_length > 0:
        Y_test_hat = test_pred
        # Y_test_hat = scaler_2.inverse_transform(test_pred)
        Y_test_real = np.array(df.iloc[-test_length:,0])
        Y_test_real = np.reshape(Y_test_real, Y_test_hat.shape)
        
    train_score = np.sqrt(mean_squared_error(Y_train_hat, Y_train_real))
    if test_length > 0:
        test_score = np.sqrt(mean_squared_error(Y_test_hat, Y_test_real))
    
    #Z is prediction of next period
    return train_pred, test_pred, train_score, test_score, Z
