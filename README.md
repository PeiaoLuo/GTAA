# 项目介绍

本项目基于对冲基金的SGTAA(Systematic Global Tactical Asset Allocation)策略，即在全球市场上以”市场“意义而非”单个“资产意义购进资产进行资产配置。SGTAA的根本理念是：市场不完全有效，通过主动投资策略可以获得超额收益。在一个或多个市场之间，通过观察市场上资产收益的走势并结合宏观经济情况，适时改变不同资产之间的配比，构建投资组合以最大程度将自身收益与市场波动分离，得到稳定收益，在此基础上尽可能得到高收益。

SGTAA和GTAA是对SAA(Strategic Asset Allocation)策略的中短期适用的调整，基于SAA战略所注重的长期的非常低频的调整结合了一定的市场时机把握，用中低频的交易谋求获得更好的收益。SGTAA与一般的GTAA的区别在于没有考虑私下与政府官员等人员访谈获取政策信息。

本项目考虑简单实现这一策略，设定的标准为，锁定具体市场不做变动，每月调整一次资产在市场间的配置比例。不考虑各个市场的交易成本特征，基于纸面最优进行每一期的配置，每一期均不保留上期的配置结果，直接调整至当期计算得到的的最优比率。不进行short操作。

***项目的具体实现、部分详细原理和回测结果见SGTAA.md文件***

# 项目结构

本项目分为

* 运行前须知和要求：README.md(本文件)，requirements.txt
* 运行部分：主文件main.py，模组modules文件夹
* 数据输入：data文件夹
* 策略来源：strategy_resource文件夹
* 结果部分：其余所有文件夹

**具体说明见各文件夹内markdown文件**

# 项目运行

将压缩文件解压缩到路径route

运行前需要确保环境(python3.6)符合requirements.txt中要求

```
route\SGTAA_prj>conda create -n test python=3.6
route\SGTAA_prj>conda activate test
```

```
(test)route\SGTAA_prj>pip install -r requirements.txt
```

本项目运行前的配置在modules/setting.py中进行修改（具体修改见setting.py中注释）

采用配置的其他默认值不改变，分别设置method='autoreg'和method='LSTM'，然后运行main.py即可得到所有目录下的结果

```
(test)route\SGTAA_prj>python main.py
```
