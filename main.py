#standard 
import warnings
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#self_defined
from modules import form, calc
warnings.filterwarnings("ignore", message="No frequency information was provided")
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.")

#make directories
directories = ['predict_plots', 'LSTM_predict_plots', 'allocation', 'back_test', 'covariance']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

#----------------------------------settings--------------------------------
from modules import setting
#if want backtest then set back_test = 1, if not, any else is ok
back_test = setting.back_test
method = setting.method

#the date of the essential allocation output, 1 for next month, 0 for this month...
date_to_predict = setting.date_to_predict

#basic settings for autoreg prediction
lag = setting.lag
trend = setting.trend
settings = [lag, trend, False]

#----------------------------------settings--------------------------------

# Chinese English comparation
asset_dict = {
    '年国债': 'bond',
    '股票': 'stock',
    '股指': 'stock_futures',
    '政府债': 'financial_futures',
    '能源': 'energy_futures',
    '肉类': 'meat_futures',
    '作物': 'grains_futures',
    '软商品': 'softs_futures',
    '金属': 'metals_futures',
    '原油': 'crude_oil_futures',
    '黄金': 'gold_futures',
    '虚拟货币': 'cryptocurrency',
    '汇率': 'exchange_rate',
    '利率': 'interest_rate',
    'PPI': 'PPI',
}
country_ls = ['China','US','UK','EU','Japan']
country_dict = {
    '美国': 'US',
    '中国': 'China',
    '日本': 'Japan',
    '德国': 'EU',
    '英国': 'UK',
    '印度': 'India',
    '巴西': 'Brazil',
    '欧盟': 'EU',
    '新加坡': 'Singapore',
    '比特币': 'BTC',
    '以太坊': 'ETH',
}

#------------------------load datas and formulate them------------------------------
print('formulate the data...')
df_ls = form.load_data()
form.rename_cols(df_ls,asset_dict,country_ls,country_dict)
form.set_range_and_type(df_ls)

#detrend df_ls by calculating changing rate
detrend_ls = calc.detrend(df_ls)

#asset_info_dict contains all assets return rate data below will use
asset_info_dict = calc.get_sep_info_dict(sep_ls=country_ls+['others'], df_ls=detrend_ls)
print('done')

#-------------------------------calculate cov matrix------------------------------
print('calculate cov matrix...')
temp_dict = form.seperate_df(df_ls, country_ls)
ordered_df = form.order_by_sep_dict(temp_dict)

if (ordered_df == 0).any().any():
    raise ValueError("There's 0 in prices!")

cov_matrix = calc.get_cov_matrix(ordered_df, date_to_predict, smooth=['Robust','SG','SMA'], window=5, poly=3, picture=True, save=True)
print('done')

#--------------------------------LSTM prediction&final allocation----------------------------------
# get the result of prediction of return rate of given date
# trend picture = True if want the LSTM prediction data in the past
# print('LSTM...')
# r_res = calc.get_predict_results(settings=settings, date_to_predict=date_to_predict, asset_info_dict=asset_info_dict, trend_picture=True, pred_picture=True, method='LSTM', LSTM_setting=[1,100])

# #formulate r_res: dict -> ret: array(values), cov_matrix: DataFrame -> cov: array
# ret = pd.DataFrame.from_dict(r_res, orient='index', columns=['return_rate'])
# ret = np.array(ret['return_rate'])
# cov = np.array(cov_matrix)

# #get final asset allocation with or without adjustment of return rate prediction
# final_res = calc.get_final_res(cov=cov)
# final_res_adjusted = calc.get_final_res(cov=cov,r=ret)
# allocation_LSTM = pd.DataFrame(data=[final_res,final_res_adjusted],columns=ordered_df.columns,index=['without adjustment','with adjustment'])
# date = ordered_df.index[-1] + pd.DateOffset(months=date_to_predict)
# date = date.strftime('%Y-%m-%d')
# allocation_LSTM.to_csv(f'allocation/LSTM_{date}.csv')
# print('done')
#---------------------------------Autoreg prediction&final allocation-------------------------------
#same as above, change the method to autoreg
print('Autoreg...')
r_res = calc.get_predict_results(settings=settings, date_to_predict=date_to_predict, asset_info_dict=asset_info_dict, trend_picture=True, pred_picture=True, method='autoreg')
ret = pd.DataFrame.from_dict(r_res, orient='index', columns=['return_rate'])
ret = np.array(ret['return_rate'])
cov = np.array(cov_matrix)
final_res = calc.get_final_res(cov=cov)
final_res_adjusted = calc.get_final_res(cov=cov,r=ret)
allocation_auto = pd.DataFrame(data=[final_res,final_res_adjusted],columns=ordered_df.columns,index=['without adjustment','with adjustment'])
date = ordered_df.index[-1] + pd.DateOffset(months=date_to_predict)
date = date.strftime('%Y-%m-%d')
allocation_auto.to_csv(f'allocation/autoreg_{date}.csv')
print('done')

#----------------------------------------back test---------------------------------
if back_test != 1:
    exit(0)
print('back test...') 
#prepare backtest data
bt_df = ordered_df.dropna(how='any')
bt_df = bt_df.pct_change()*100
bt_df.dropna(how='any',inplace=True)

max_back = 18
window_size = 6  # For SMA
alpha = 0.5  # For EMA

ret_ls = []
ret_adjusted_ls = []

#body of back test
for i in range(max_back):
    date_to_predict = 0 - i
    cov_matrix = calc.get_cov_matrix(ordered_df, date_to_predict, smooth=['Robust','SG','SMA',], window=5, poly=3)
    #trend and predict picture alaways false here
    r_res = calc.get_predict_results(settings=settings, date_to_predict=date_to_predict, asset_info_dict=asset_info_dict, method=method, LSTM_setting=[1,100])
    
    ret = pd.DataFrame.from_dict(r_res, orient='index', columns=['return_rate'])
    ret = np.array(ret['return_rate'])
    cov = np.array(cov_matrix)

    final_res = calc.get_final_res(cov=cov)
    final_res_adjusted = calc.get_final_res(cov=cov,r=ret)
    
    final_ret = sum(final_res * bt_df.iloc[-2 + date_to_predict,:])
    final_ret_adjusted = sum(final_res_adjusted * bt_df.iloc[-2 + date_to_predict,:])
    
    print(f"{i+1}/{max_back}: R: {final_ret} | R_a: {final_ret_adjusted}")
    
    ret_ls.append(final_ret)
    ret_adjusted_ls.append(final_ret_adjusted)
    
#choose how long back the plot will show
date_length = 17
ret_ls = ret_ls[1:1+date_length]
ret_adjusted_ls = ret_adjusted_ls[1:1+date_length]

#get value: prod result
rets = []
a_rets = []

for i in range(len(ret_ls)):
    ls = ret_ls[i:]
    new_ret = pd.Series(ls)
    ret_arr = np.array(new_ret)/100 + 1
    rets.append(ret_arr.prod())

for i in range(len(ret_adjusted_ls)):
    ls = ret_adjusted_ls[i:]
    new_ret = pd.Series(ls)
    ret_arr = np.array(new_ret)/100 + 1
    a_rets.append(ret_arr.prod())
    
average_return = np.array(ret_ls).prod()**(1/(len(ret_ls)-1))
adjusted_average_return = np.array(ret_adjusted_ls).prod()**(1/(len(ret_adjusted_ls)-1))
rets += [1]
a_rets += [1]
rets.reverse()
a_rets.reverse()
with open('history_net_value.txt', 'w') as fp:
    for item in rets:
        fp.write(f"{item}, ")
    fp.write('\n')
    for item in a_rets:
        fp.write(f"{item}, ")
#sharpe ratio
pct_rets = [x * 0.01 for x in ret_adjusted_ls]
std_ret_adjusted = np.std(pct_rets, ddof=1)
annual_std_ret_adjusted = std_ret_adjusted*np.sqrt(12)
annual_ret = (1+0.01*adjusted_average_return)**12 - 1
risk_free_ret = 0.0153

sharpe_ratio = (annual_ret - risk_free_ret)/annual_std_ret_adjusted

#max drawdown
peaks = np.maximum.accumulate(a_rets)
drawdowns = 1 - (a_rets / peaks)
max_drawdown = np.max(drawdowns)

#record
with open(f'back_test/{method}_{date}port_avg_return_rate.txt', 'w') as fp:
    fp.write('without adjustment:\n')
    for item in ret_ls:
        fp.write(f"{item},")
    fp.write('\n')
    fp.write(f'average: {average_return}\n')
    fp.write('\n')
    fp.write('with adjustment:\n')
    for item in ret_adjusted_ls:
        fp.write(f"{item},")
    fp.write('\n')
    fp.write(f'average: {adjusted_average_return}\n')
    fp.write('\n')
    fp.write(f'sharpe_ratio: {sharpe_ratio}\n')
    fp.write(f'max drawdown: {max_drawdown}\n')

#----------------------------bt_plot----------------------------
#return rate
plt.plot(ret_ls, label='ret')
plt.plot(ret_adjusted_ls, label='ret_a')
plt.xlabel('Date')
plt.ylabel('Return rate')
plt.title('portfolio_return_rate')
plt.legend()
plt.savefig(f'back_test/{method}_portfolio_return_rate.png')
plt.close()

#value

plt.plot(rets, label='prod_ret')
plt.plot(a_rets, label='pord_ret_a')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('portfolio_value')
plt.legend()
plt.savefig(f'back_test/{method}_portfolio_value.png')
plt.close()
print('done')