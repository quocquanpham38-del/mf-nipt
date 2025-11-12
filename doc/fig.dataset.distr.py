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


# 使用huggingface的dataset库读取数据
repo_id = "sxj1024/cumcm_test"
dataset = load_dataset(repo_id)
data = dataset['train'].to_pandas()  # 转换为pandas DataFrame

# 数据筛选和提取
female_fetus_data = data[data[' Fetal Type (e.g., Male Fetus)'] == 'Female Fetus']
male_fetus_data = data[data[' Fetal Type (e.g., Male Fetus)'] == 'Male Fetus']

x_chromosome_z_score_female = female_fetus_data[' Z-Score of Chromosome X']
y_chromosome_concentration_male = male_fetus_data['Concentration of Chromosome Y']
bmi_data = data[" Pregnant Woman's BMI (BMI: Body Mass Index)"]

# 创建画布
plt.subplots(2, 3, figsize=(12, 8))


# 第一行：直方图（3个子图）
# 1. 女性胎儿X染色体Z值直方图
plt.subplot(2, 3, 1)
plt.hist(x_chromosome_z_score_female, bins=20, edgecolor='black')
plt.xlabel('X-Chromosome Z-Score')
plt.ylabel('Female Fetus Samples')

# 2. 男性胎儿Y染色体浓度直方图
plt.subplot(2, 3, 2)
plt.hist(y_chromosome_concentration_male, bins=20, edgecolor='black')
plt.xlabel('Y-Chromosome Concentration')
plt.ylabel('Male Fetus Samples')

# 3. 母体BMI直方图
plt.subplot(2, 3, 3)
plt.hist(bmi_data, bins=20, edgecolor='black')
plt.xlabel('BMI')
plt.ylabel('Maternal samples')

# 1. 女性胎儿X染色体Z值小提琴图
plt.subplot(2, 3, 4)
axes = plt.gcf().get_axes()
sns.violinplot(y=x_chromosome_z_score_female, ax=axes[3])
plt.ylabel('X Chromosome Z-Score')

# 2. 男性胎儿Y染色体浓度小提琴图
plt.subplot(2, 3, 5)
sns.violinplot(y=y_chromosome_concentration_male, ax=axes[4])
plt.ylabel('Y Chromosome Concentration')

# 3. 母体BMI小提琴图
plt.subplot(2, 3, 6)
axes = plt.gcf().get_axes()
sns.violinplot(y=bmi_data, ax=axes[5])
plt.ylabel('BMI')


# 调整布局并保存
plt.tight_layout()
plt.savefig('./doc/fig.dataset.distr.svg')
plt.savefig('./doc/fig.dataset.distr.jpg', dpi=400)
plt.show()
