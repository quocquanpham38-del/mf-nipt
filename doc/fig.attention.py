import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置图片清晰度和字体
import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/font/times/times.ttf')
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 14})# 统一设置所有字体大小为14
plt.rcParams['figure.dpi'] = 400

x_labels=['t', 'A', 'BMI', 'W', 'X','Y', '13', '18', '21']
y_labels=['21-High', '18-High','13-High', '21-Low', '18-Low', '13-Low', '21-Indeterminate', '18-Indeterminate', '13-Indeterminate']

attn = [[0.200292,  0.838234,   0.704412,   0.57758,    0.251835,   0.249997,   0.28098,    0.179115,   0.898263,],
        [0.142023,  0.752699,   0.728459,   0.63879,    0.158695,   0.161776,   0.161308,   0.810829,   0.260542,],
        [0.146899,  0.878641,   0.63108,    0.50879,    0.148659,   0.12921,    0.876031,   0.378614,   0.201329,],
        [0.1426478, 0.807773,   0.630255,   0.73542,    0.210727,   0.192539,   0.13118,    0.063902,   0.713659,],
        [0.109087,  0.749373,   0.540387,   0.605064,   0.155677,   0.116206,   0.182462,   0.749639,   0.245263,],
        [0.032313,  0.659829,   0.500819,   0.285275,   0.162058,   0.081599,   0.756471,   0.254094,   0.187306,],
        [0.742023,  0.152699,   0.328459,   0.43879,    0.358695,   0.361776,   0.061308,   0.110829,   0.660542,],
        [0.785346,  0.12608,    0.487875,   0.344924,   0.323912,   0.455514,   0.068919,   0.53291,    0.119378,],
        [0.794946,  0.024857,   0.472794,   0.362546,   0.20587,    0.328415,   0.688675,   0.154815,   0.047081,]
        ]
attn = np.array(attn)

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(attn, aspect='auto', interpolation='nearest', cmap='viridis')

# 设置坐标轴刻度与标签
ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

# 旋转 x 标签，便于显示
# plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# 在每个格子里注释数值（保留两位小数）
for i in range(attn.shape[0]):
    for j in range(attn.shape[1]):
        text = ax.text(j, i, f"{attn[i, j]:.2f}",
                       ha="center", va="center", fontsize=8, color="w" if attn[i,j] < 0.5 else "black")

# 添加色条
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Attention score', rotation=-90, va="bottom")

# 标题与布局调整
ax.set_xlabel('Factors')
ax.set_title('Attention scores heatmap')


# 调整布局并保存
plt.tight_layout()
plt.savefig('./doc/fig.attention.svg')
plt.savefig('./doc/fig.attention.jpg', dpi=400)
plt.show()