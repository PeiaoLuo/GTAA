### 策略简略描述

与本对冲的投资目的相符，拟构建投资组合以最大程度将自身收益与市场波动分离，得到稳定收益，在此基础上尽可能得到高收益。

本基金拟在宏观上进行资产配置，即对市场内的资产以整体进行投资而非筛选单个资产；本基金的投资组合再平衡规则为，投资的具体对象固定，根据数据更新、模型变化、表现情况等，每月调整其比率。

步骤为：

1.对以地区（中、美、英、德、日、其他）、资产类型（股票、国债、期货、黄金、原油、加密数字货币）为划分的46个资产计算协方差矩阵，据此采用风险平价理论进行初步配资，此初步配资应当对市场的风险暴露降到低值，满足SGTAA的收益与市场高度无关的要求

    风险平价理论：不同资产类别在整体风险上对投资组合的贡献应该相等，具体见下文实现部分

2.对此些资产应用autoreg、LSTM两者分别进行收益率预测，根据预测结果对上述配资结果进行调整，基于对收益率预测的判断，略微增大投资组合的风险暴露，以求得到更高收益，最后得到两个结果

3.依上述方式配资进行回测

注：此策略没有考虑交易成本影响，交易策略为每月根据计算结果将资产完全调整至纸面最优配置水平，该策略不进行short，所有资产配置比率均不为负

### 具体实现

#### 1.数据加载和格式调整（main.py，line67）（form.py/calc.py，line21）

原始数据频率为月度，值为指数值或者价格（以该资产所在地区货币计价）

此部分对数据值的调整只有采用百分比变动去趋势得到月度收益率，其余均是结构调整，略

考虑到SGTAA策略在全球范围内广泛配资，不再将汇率因素剔除，然后再单独考虑，采用原始货币计算同时考虑市场间相关性和汇率的影响

#### 2.协方差矩阵计算（main.py，line80）（calc.py，line31↓）

对所有资产指数或价格百分比变动数据采用：

origin_data——>RobustScaler——>savgol_filter(window=5,polyorder=3)——>SMA(window=6)——>processed_data

* RobustScaler进行离群值处理，保留离群值且不会让它影响过大，有助于捕捉特别事件的影响
* Savitzky-Golay滤波器去噪，特别针对周期性和趋势进行去噪
* SAM简单移动平均平滑，进一步消除短期噪音和随机波动，凸显趋势

通过此过程以减轻掺杂大量信息的指数或价格信息中无关部分，以提高协方差的有效性

采用processed_data计算协方差矩阵cov（**结果在covariance文件夹中**）

#### 3.下期收益率计算（main.py，line91）（calc.py，line96↓；LSTM.py）

比对了进行类似协方差的预处理与否的效果（不进行预处理），并考虑了市场结构变化（信息的时效性，只往后选72期），最终不进行任何预处理直接采用当期到前72期的数据训练模型预测下一期的收益率，分别采用以下两种模型

* **Autoreg**

```
autoreg(lags=4,trend='c',exog=[PPI,CBR(央行基准利率),NEER(名义有效汇率)],seasonal=False)
```

$$
y_t = AY_t + BX + C + \epsilon_t
$$

$$
Y_t = [y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4}]
$$

$$
X=[PPI_{t-1},CBR_{t-1},NEER_{t-1}]
$$

    采用自回归模型进行预测，设定为：滞后4期，常数趋势，无季节性，取当地前一年PPI、央行基准利率、名义有效汇率为外生变量

    分别对46个资产应用此模型进行训练得到下期收益率，并绘制这些模型预测的历史曲线（**图片在predict_plots文件夹中**）

* **LSTM**

```
model= Sequential()
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

    采用线性堆叠的神经网络，添加50个单元的LSTM层，添加全连接输出为单值的输出层，模型训练的损失函数设置为MSE，采用Adam进行模型参数优化

    分别对46个资产应用此模型进行训练得到下期收益率，并绘制这些模型预测的历史曲线（**图片在LSTM_predict_plots文件夹中**）

#### 4.最终配置（main.py，line91）（calc.py，line255↓）

采用风险平价理论进行资产的初步配置

**风险平价**：

让各个资产的加权风险边际贡献相等，以减小资产组合的风险暴露

目标函数设置为risk_budget_objective(weights,cov)，约束函数设置为total_weight_constraint(x)，二者共同表达：

---

$$
\text{Minimize} \quad \sum_{i,j=1}^{N} (TRC_i - TRC_j)^2
$$

$$
\text{TRC}_i = w_i \cdot \text{MRC}_i
$$

$$
\text{MRC}_i = \frac{\sum_{j=1}^{N} \text{Cov}_{ij} \cdot w_j}{\sqrt{\sum_{k=1}^{N} \sum_{l=1}^{N} \text{Cov}_{kl} \cdot w_k \cdot w_l}}
$$

subject to:

$$
w_i > 0 \quad, 
\sum_{i=1}^{N} w_i = 1
$$

其中

$$
weights = [w_{1},w_{2},...,w_{N}]
$$

$$
Cov：上面步骤得到的协方差矩阵
$$

$$
MRC：风险边际贡献，TRC：加权风险边际贡献
$$

采用序列最小二乘法实现该目标函数的求解，各资产初始设置为平均配资，用SLSQP方法实现为：

```
x0 = np.ones(cov.shape[0]) / cov.shape[0]
bnds = tuple((0,1) for x in x0)
cons = ({'type': 'eq', 'fun': total_weight_constraint})
options={'disp':False, 'maxiter':1000, 'ftol':1e-20} 
solution = minimize(risk_budget_objective, x0, args=(cov), bounds=bnds, constraints=cons, method='SLSQP', options=options)
```

获得初步结果solution.x以后，简单的进行风险暴露调整：下一期收益率最大的资产持有比率绝对数值增加5%，其余所有资产持有比率减少至原来的95%，得到最后配置（**不同方法组合的配置在allocation文件夹中**）

#### 5.回测（main.py，line125）

照此方案配置资产，回测过去18期的收益率，获得基于autoreg、LSTM调整的或者不调整的配置方案的累计净值图片和平均月复利利率以及各种评价标准（**back_test文件夹中**）

#### 6.结果

经过比对，采用autoreg进行收益率预测，采用收益率进行调整的策略有最大的月复利利率和累计净值，最后采用此种方法进行配置

结果为：

**月复利利率:1.62%**

**从18月前为1，到最近的累计净值:1.314**

**shape ratio: 2.62 （无风险收益率采用1.53%：4-30的3个月国债收益率）**

**最大回撤: 0.022**
