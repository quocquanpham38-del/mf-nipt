import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# 设置图片清晰度和字体
import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 14})# 统一设置所有字体大小为14
plt.rcParams['figure.dpi'] = 400

cov=[10.265,6.868,3.288,4.285,4.986,3.91,3.847,3.303,3.161,2.386,
     3.217,2.311,2.746,2.297,1.272,1.079,1.872,1.943,1.537,1.551,1.1000,
     1.051,0.761,0.788,0.716,0.744,0.558,0.293,0.42,1.018,0.352,0.317,
     0.124,0.114,0.101,0.108,0.094,0.072,0.03,0.035,0.05,0.044,0.068,
     0.01,0.008,0.009,0.006,0.005,0.01,0.009,0.007,0.008,0.007,0.006,
     0.005,0.005,0.004,0.002,0.004,0.003,0.003,0.003,0.003,0.004,0.003,
     0.002,0.0018,0.0011,0.0014,0.0012,0.0013,0.0016,0.0016,0.0018,0.0015,0.002,
     0.001,0.000638,0.000954,0.000558,0.00061,0.000973,0.000953,0.000669,0.000947,0.00071,0.000743,
     0.0005,0.000311,0.000364,0.000464,0.000344,0.000335,0.000364,0.000456,0.000366,0.000358,0.00034,
     0.0003,0.000235,0.000149,0.000107,0.000226,0.000291,0.000141,0.000212,0.000246,0.000169,0.000241,
     0.0001,0.000075,0.000082,0.000063,0.000072,0.00009,0.000078,0.00007,0.000065,0.000068,0.00006]
plt.figure(figsize=(5,4))

plt.plot(cov,color="skyblue",linewidth=2)
plt.xlabel('Training Epoch')
plt.ylabel('Loss')
plt.xticks(ticks=range(0,111,10), labels=range(0,12,1))

# 调整布局并保存
plt.tight_layout()
plt.savefig('./doc/fig.convergennce.svg')
plt.savefig('./doc/fig.convergence.jpg', dpi=400)
plt.show()