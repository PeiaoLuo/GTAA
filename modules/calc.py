import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from modules import setting
back = setting.back

def get_time_range(df_ls) -> dict:
    time_range = {}
    for df in df_ls:
        for col in df.columns:
            st = df[col].first_valid_index()
            ed = df[col].last_valid_index()
            time_range[col] = (st, ed)
    return time_range

#detrend data into pct_change
def detrend(df_ls) -> list:
    detrend_df_ls = []
    for df in df_ls:
        mask = df != 0
        pct_changes = df[mask].pct_change()*100
        detrend_df = df.copy()
        detrend_df[mask] = pct_changes
        detrend_df_ls.append(detrend_df)
    return detrend_df_ls

#----------------------------------------------covariance matrix----------------------------------------------
#calculate covariance matrix
def get_cov_matrix(df, date_to_predict, smooth=[None], window=3, poly=2, picture=False, save=False) -> pd.DataFrame: # get cov matrix of 
    #cut data: the date range is first month to date_to_predict (1=next, 0=now, -1=last, etc.)
    ed = -1 + date_to_predict + len(df)
    if ed == len(df):
        ed = ed - 1
    if ed > back:
        df = df.iloc[ed - back:ed,:]
    else:
        df = df.iloc[:ed,:]
    
    for method in smooth:
    
        #standardization
        if method == None:
            pass
        elif method == "Robust":
            robust_scaler = RobustScaler()
            std_df = robust_scaler.fit_transform(df)
            std_df = pd.DataFrame(std_df, columns=df.columns)
            df = std_df
        elif method == "Minmax":
            robust_scaler = MinMaxScaler()
            std_df = robust_scaler.fit_transform(df)
            std_df = pd.DataFrame(std_df, columns=df.columns)
            df = std_df
        
        # smoothing
        elif method == 'SG':
            temp_dict = {}
            for col in df.columns:
                temp_dict[col] = savgol_filter(df[col], window_length=window, polyorder=poly)
            df = pd.DataFrame.from_dict(temp_dict)        
        elif method == 'WT':
            pass
        elif method == 'SMA':
            smoothed_df = df.rolling(window=6, min_periods=1).mean()
            df = smoothed_df
        elif method == 'EMA':
            smoothed_df = df.ewm(alpha=0.5, adjust=False).mean()
            df = smoothed_df
        else:
            pass
    
    cov_matrix = df.cov()
    
    if picture:
        import matplotlib.pyplot as plt
        import seaborn as sns 
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        norm = plt.Normalize(vmin=cov_matrix.values.min(), vmax=cov_matrix.values.max())
        plt.figure(figsize=(18, 15))
        sns.heatmap(cov_matrix, cmap=cmap, center=0, annot=False, fmt=".2f", linewidths=0.5)
        plt.title('Covariance Matrix')
        plt.xlabel('Variables')
        plt.ylabel('Variables')
        plt.savefig("covariance/cov.png")
        plt.close()
    
    if save:
        cov_matrix.to_csv('covariance/covariance_matrix.csv')
    
    return cov_matrix

#--------------------------------------Autoreg/LSTM Prediction-------------------------------------------------
#process the initially loaded data: df_ls into structure fits later prediction use
def get_sep_info_dict(sep_ls, df_ls) -> dict: #separate from df_ls into dict whose keys are
    #sep_ls, into {'AreaA':[datas], 'AreaB':[datas],}
    #where datas = [dataframe({'asset_name':[data], 'PPI':[data], ...}), ...]
    #areas without PPI etc. datas will be datas = [series({'asset_name':[data]}), ...]
    asset_info_dict = {}
    for i,area in enumerate(sep_ls):
        temp_ls = []
        if area != 'others':
            # seperate into col + PPI + Exchange + Interest
            for j in range(len(df_ls[i].columns)-3):
                temp_df = df_ls[i].iloc[:,[j,-3,-2,-1]].copy()
                temp_df.dropna(how='any',inplace=True)
                temp_df.replace([np.inf], 0.0, inplace=True)
                temp_ls.append(temp_df)
        else:
            # seperate into single col
            for j in range(len(df_ls[i].columns)):
                temp_df = pd.DataFrame(df_ls[i].iloc[:,j].copy())
                temp_df.dropna(how='any',inplace=True)
                temp_df.replace([np.inf], 0.0, inplace=True)
                temp_ls.append(temp_df)
        asset_info_dict[area] = temp_ls
    return asset_info_dict

#get Autoreg\LSTM prediction of the return rate, the plots are also generated here
def get_predict_results(settings: list, date_to_predict, asset_info_dict, evaluation=True, trend_picture=False, pred_picture=False, method='autoreg', LSTM_setting=[1,50]) -> dict:
    #trend_picture: draw trend picture for all assets
    #pred_picture: draw picture for r of date_to_predict
    if (evaluation and trend_picture) or pred_picture:
        import matplotlib.pyplot as plt
        import seaborn as sns
    
    res_dict = {}
    
    lag = settings[0]
    trend = settings[1]
    seasonal = settings[2]
    
    for asset_df_name,value in asset_info_dict.items():

        for dt in value:
            backward = -1 + date_to_predict + len(dt)
            #testdt is the training datas
            if asset_df_name == 'others':
                if backward > back:
                    testdt = dt[backward - back:backward]
                else:
                    testdt = dt[:backward]
            else:
                if backward > back:
                    testdt = dt.iloc[backward - back:backward,:]
                else:
                    testdt = dt.iloc[:backward,:]
            
            # testdt = de_noise(testdt, ['Robust','SG','SMA',])
            if method == 'autoreg':
                if asset_df_name == 'others':
                    model = AutoReg(testdt, lags=lag, trend=trend, seasonal=seasonal)
                else:
                    model = AutoReg(testdt.iloc[:,0], lags=lag, trend=trend, exog=testdt.iloc[:,1:], seasonal=seasonal)
                #get result
                res = model.fit()
                
                #here's calculate the next month's return
            
                if asset_df_name != 'others':
                    lag_dt = [dt.iloc[backward-1-i,0] for i in range(lag)]
                    other_dt = list(dt.iloc[backward-1,1:])
                else:
                    lag_dt = [dt.iloc[backward-1-i, 0] for i in range(lag)]
                    other_dt = []
                    
                flatten_values = np.array([1] + lag_dt + other_dt)

                params = np.array(res.params)
                
                predict_next_m = np.dot(params, flatten_values)
                
                res_dict[dt.columns[0]] = predict_next_m
            
            elif method == 'LSTM':
                from modules import LSTM
                train_pred, test_pred, train_score, test_score, predict_next_m = LSTM.train(testdt, test_length=1, back=4, settings=LSTM_setting)
                res_dict[dt.columns[0]] = predict_next_m[0][0]
                if (evaluation and trend_picture):
                    pred = np.concatenate([train_pred, test_pred], axis=0).flatten()
                    plt.figure(figsize=(20,6))
                    sns.lineplot(x=testdt.index[4+1:], y=testdt.iloc[4+1:,0], label=dt.columns[0])
                    sns.lineplot(x=testdt.index[4+1:], y=pred, label='Predicted')
                    plt.xlabel('Date')
                    plt.ylabel('Return rate')
                    plt.title(f'{dt.columns[0]}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'LSTM_predict_plots/{testdt.columns[0]}.png')
                    plt.close()
            else:
                raise ValueError(f'undefined method for prediction of return rate: {method}')
            
            #evaluation
            if (evaluation and method == 'autoreg'):
                if asset_df_name == 'others':
                    predictions = res.predict(start=testdt.index[lag], end=testdt.index[-1])
                    predictions_df = pd.DataFrame(predictions, index=testdt.index[lag:], columns=['Predicted'])
                    combined_df = pd.concat([testdt, predictions_df], axis=1)
                    
                    plot_name = combined_df.columns[0]
                    combined_df.rename(columns={combined_df.columns[0]: 'Real'})
                    
                    combined_df.reset_index(inplace=True)

                    tidy_df = combined_df.melt(id_vars='index', var_name='Variable', value_name='Value')
                    
                else:
                    predictions = res.predict(start=testdt.index[lag], end=testdt.index[-1], exog=testdt.iloc[:, 1:])
                    predictions_df = pd.DataFrame(predictions, index=testdt.index[lag:], columns=['Predicted'])
                    combined_df = pd.concat([testdt.iloc[:, 0], predictions_df], axis=1)
                    
                    plot_name = combined_df.columns[0]
                    combined_df.rename(columns={combined_df.columns[0]: 'Real'})
                    
                    combined_df.reset_index(inplace=True)

                    #tidy_df contains the real val and the predict val, further analysis like MSE can be directly derived from tidy_df
                    tidy_df = combined_df.melt(id_vars='index', var_name='Variable', value_name='Value')
                
            #here's drawing part
            
            if evaluation and trend_picture and method == 'autoreg':
                plt.figure(figsize=(20, 6))
                sns.lineplot(data=tidy_df, x='index', y='Value', hue='Variable')
                plt.xlabel('Date')

                plt.ylabel('Return rate')
                plt.title(f'{plot_name}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"predict_plots/{plot_name}.png")
                plt.close() 
                
    if pred_picture:
        assets = list(res_dict.keys())
        bar_lengths = list(res_dict.values())
        
        plt.figure(figsize=(18, 12))
        plt.barh(assets, bar_lengths, color='skyblue')
        plt.xlabel('Returns(%)')
        plt.ylabel('Assets')
        plt.title('Next Month\'s Return Rate')
        if method == 'LSTM':
            plt.savefig("allocation/LSTM_return_rate.png")
        else:
            plt.savefig("allocation/autoreg_return_rate.png")
        plt.close()
        
    return res_dict

#-------------------------------------Risk Parity-------------------------------------
#target function of Risk Parity Theory
def risk_budget_objective(weights,cov):
    weights = np.array(weights)
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))

    MRC = np.dot(cov,weights)/sigma 
    TRC = weights * MRC
    
    #delta is the minization target
    delta_TRC = [sum((i - TRC)**2) for i in TRC]
    return sum(delta_TRC)

#constraint of weright
def total_weight_constraint(x):
    return np.sum(x)-1.0

def get_diagonal_elements(matrix):
    n = len(matrix)  
    diagonal_elements = []
    for i in range(n):
        diagonal_elements.append(matrix[i][i])
    return diagonal_elements

#based on the Risk Parity Theory and predicted return of next period, get the final asset allocation
def get_final_res(cov, r=None) -> list: #without passing r, the diversification will
    #purely based on cov, with passing r, the diversification will based on cov and r
    x0 = np.ones(cov.shape[0]) / cov.shape[0]
    bnds = tuple((0,1) for x in x0)
    cons = ({'type': 'eq', 'fun': total_weight_constraint})
    options={'disp':False, 'maxiter':1000, 'ftol':1e-20} 

    if r is None:
        solution = minimize(risk_budget_objective, x0, args=(cov), bounds=bnds, constraints=cons, method='SLSQP', options=options)
        weights = solution.x
    else:
        solution = minimize(risk_budget_objective, x0, args=(cov), bounds=bnds, constraints=cons, method='SLSQP', options=options)
        variances = get_diagonal_elements(cov)
        # r = r/np.sqrt(variances)
        max_index = np.argmax(r)
        weights = solution.x
        #simple adjustment with r
        for i in range(len(weights)):
            if i != max_index:
                weights[i] *= 0.95
        weights[max_index] += 0.05
        
    return weights
