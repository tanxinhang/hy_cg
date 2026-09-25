# WLS-MAP 与队形 baseline pilot

## 结论

- Jacobian/WLS-MAP 达到低复杂度目标，但只适合作为局部线性 baseline；150 m
  偏差下不能替代多峰位置搜索。
- 20 m 约束下，单视角、双视角和 belief-aware 队形都显示正向几何收益；完整
  TP-UIC pilot 中双视角的中位最差目标信息最高，当前 belief-aware 几何 proxy
  尚未稳定支配双视角。

## Jacobian/WLS-MAP

实现位于 `isac_sim/receiver/target_state/wls.py`。它在白化后的中心/切向目标字典
上拟合复系数，由切向系数与中心系数之比得到每条双基地链路的 DD 创新，再用
现有 delay/Doppler Jacobian 与位置先验做二维 WLS-MAP。最终只在估计点和先验
中心做精确目标函数评价。

150 m、三个配对场景的一折状态结果：WLS误差 126.1/151.9/144.0 m，耗时
0.34/0.36/0.37 s；gated_topk误差 16.7/2.0/5.6 m，耗时
15.89/16.87/16.16 s。WLS约快45倍，但大偏差下线性化失效。

## 队形

20个场景的纯 belief 几何筛选中，相对随机队形的最差目标 proxy 中位数，单视角、
双视角、belief-aware 分别约为 5.3、14.6、15.4 倍。该指标只用于筛选。

随后在3个配对场景上运行完整 TP-UIC，并将六接收机的目标邻域 `ncp_unit` 求和。
最差目标信息的中位数为：随机 210.2、单视角 247.8、双视角 394.1、
belief-aware 284.9。三个场景太少，尚不能换算成正式 P_D 结论。

所有生成队形均满足三维节点间距不小于20 m，移动上限为200 m。
