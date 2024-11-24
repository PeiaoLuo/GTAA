**该文件夹是回测的结果，回测为从有数据的最新一期往回18期（每期为一个月）经过回测比较最终选择的策略是autoreg调整的配置**

autoreg开头为使用autoreg预测收益率的结果，LSTM开头为使用LSTM

.txt文件为从日期所示期往前18期的平均复利收益率，其中的without adjustment表示没有采用预测收益率调整配资的结果，with adjustment表述采用了预测收益率调整配资的结果

shapre ratio 和 max drawdown 采用 with adjustment的值计算

.png文件

以return_rate结尾的是回测期间内每期的收益率，单位为%，ret_a是采用收益率调整的结果，ret是没有采用收益率调整的结果

以value结尾的是累计净值，单位为1。时间随增长趋势方向增加。
