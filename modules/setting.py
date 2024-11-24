#other settings like seasonal, smoothing method, draw plot or not, and other details have gone
#through tests to choose, not changable here, but can change in the code

#necessary settings
#influence the output of allocation, plots, covariance
date_to_predict = 1
#how long back(months) the data will be used in prediction
back = 72
#settings of autoreg
lag = 4 #lag periods
trend = 'c' #trend type, constant, 


#back test settings
#if want back_test set 1, if not, 0
back_test = 1
#method used for prediction in back test, autoreg or LSTM
method = 'autoreg'