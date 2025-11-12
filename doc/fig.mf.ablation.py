import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import numpy as np

# 设置图片清晰度和字体
import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 14})# 统一设置所有字体大小为14
plt.rcParams['figure.dpi'] = 400


# FF: t +BMI +H +W
FF_labels = ['t', '+BMI', '+H', '+W']
# FF_Acc = [0.814, 0.899, 0.900, 0.912]
# FF_F1  = [0.714, 0.829, 0.832, 0.868]
FF = [[0.814,0.714],
        [0.899,0.829],
        [0.900,0.832],
        [0.912,0.868]]
FF=np.array(FF)
FF_x=np.arange(len(FF_labels))
# 4个颜色列表
FF_colors = ['#1f77b4',  # 蓝
          '#ff7f0e',  # 橙
          '#2ca02c',  # 绿
          '#d62728']  # 红



# FA: t +A +BMI  +W +X +13 +18 +21
FA_labels = ['t', '+A', '+BMI', '+W', '+X', '+13', '+18', '+21']
# FA_Acc = [0.867, 0.900, 0.910, 0.915, 0.919, 0.932, 0.940, 0.957]
# FA_F1  = [0.804, 0.820, 0.829, 0.833, 0.875, 0.902, 0.913, 0.923]
FA=[[0.867,0.804],
    [0.900,0.820],
    [0.910,0.829], 
    [0.915,0.833],
    [0.919,0.875],
    [0.932,0.902],
    [0.940,0.913],
    [0.957,0.923]]
FA=np.array(FA)
FA_x=np.arange(len(FA_labels))
FA_colors = ['#1f77b4',  # 蓝
          '#ff7f0e',  # 橙
          '#2ca02c',  # 绿
          '#d62728',  # 红
          '#9467bd',  # 紫
          '#8c564b',  # 棕
          '#e377c2',  # 粉
          '#7f7f7f']  # 灰


plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
width=0.1
group_width = 6*width
for i in range(len(FF_labels)):
    plt.bar([i*width, group_width+i*width], FF[i], width, label=FF_labels[i],color=FF_colors[i])
plt.ylim(0.7, 1.0)
plt.xticks(ticks=[0.1,0.75], labels=['Accuracy','F1'])
plt.title("Ablation of NIPT time-point selection")
plt.xlabel('(a)')
plt.legend(fontsize=14,ncols=4, loc='upper center')
plt.grid(axis='y')

plt.subplot(1,2,2)
width=0.1
group_width = 10*width
for i in range(len(FA_labels)):
    plt.bar([i*width, group_width+i*width], FA[i], width, label=FA_labels[i])
plt.ylim(0.75, 1.0)
plt.xticks(ticks=[0.375,1.45], labels=['Accuracy','F1'])
plt.title('Ablation of fetal abnormal testing')
plt.xlabel('(b)')
plt.legend(fontsize=14,ncols=4, loc='upper center')
plt.grid(axis='y')

# 调整布局并保存
plt.tight_layout()
plt.savefig('./doc/fig.mf.ablation.svg')
plt.savefig('./doc/fig.mf.ablation.jpg', dpi=400)
plt.show()

