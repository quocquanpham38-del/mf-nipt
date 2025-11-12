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
week = [7,8,9,10,11,12,13,14]
FF=[2,3,3.5,4.0,5,6,6.5,7.0]
F1=[[0.845,0.876,0.910,0.923,0.923,0.923,0.923,0.923], ## MF-NIPT
    [0.754,0.823,0.891,0.915,0.915,0.915,0.915,0.915],  ## MPF
    [0.699,0.704,0.712,0.723,0.723,0.723,0.723,0.723],  ## Bayesian Inference 
    ]

plt.figure(figsize=(8,4))
plt.plot(week, F1[0],marker='o',label='MF-NIPT')
plt.plot(week, F1[1],marker='o',label='MPF')
plt.plot(week, F1[2],marker='o',label='Bayesian Inference')
plt.xlabel('Gestational week (t)')
plt.ylabel('F1')
plt.ylim(0.6,1.0)
plt.legend(fontsize=10, loc='upper center', ncols=3)
for i in range(len(week)):
    plt.text(week[i], F1[0][i]+0.01, f'FF:{FF[i]}%', fontsize=10, color='black', ha='center')
plt.grid()


# 调整布局并保存
plt.tight_layout()
plt.savefig('./doc/fig.casestudy.svg')
plt.savefig('./doc/fig.casestudy.jpg', dpi=400)
plt.show()