import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# 设置图片清晰度和字体
plt.rcParams['figure.dpi'] = 400
plt.rcParams.update({'font.size': 14})  # 统一设置所有字体大小为14

# 单位转换（800x300毫米转换为英寸）
width_mm = 800
height_mm = 300
width_inch = width_mm / 25.4
height_inch = height_mm / 25.4

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
plt.figure(figsize=(width_inch, height_inch))

# 设置网格布局：2行3列
grid = plt.GridSpec(2, 3, wspace=0.3, hspace=0.5)

# 第一行：直方图（3个子图）
# 1. 女性胎儿X染色体Z值直方图
ax1 = plt.subplot(grid[0, 0])
ax1.hist(x_chromosome_z_score_female, bins=20, edgecolor='black')
ax1.set_xlabel('X Chromosome Z-Score')
ax1.set_ylabel('Female Fetus Samples')
ax1.tick_params(axis='both', rotation=0)

# 2. 男性胎儿Y染色体浓度直方图
ax2 = plt.subplot(grid[0, 1])
ax2.hist(y_chromosome_concentration_male, bins=20, edgecolor='black')
ax2.set_xlabel('Y Chromosome cfDNA Concentration')
ax2.set_ylabel('Male Fetus Samples')
ax2.tick_params(axis='both', rotation=0)

# 3. 母体BMI直方图
ax3 = plt.subplot(grid[0, 2])
ax3.hist(bmi_data, bins=20, edgecolor='black')
ax3.set_xlabel('BMI')
ax3.set_ylabel('Maternal samples')
ax3.tick_params(axis='both', rotation=0)

# 第二行：小提琴图（3个子图）
sns.set_style("whitegrid")

# 1. 女性胎儿X染色体Z值小提琴图
ax4 = plt.subplot(grid[1, 0])
sns.violinplot(y=x_chromosome_z_score_female, ax=ax4)
ax4.set_ylabel('X Chromosome Z-Score')
ax4.tick_params(axis='both', rotation=0)

# 2. 男性胎儿Y染色体浓度小提琴图
ax5 = plt.subplot(grid[1, 1])
sns.violinplot(y=y_chromosome_concentration_male, ax=ax5)
ax5.set_ylabel('Y Chromosome Concentration')
ax5.tick_params(axis='both', rotation=0)

# 3. 母体BMI小提琴图
ax6 = plt.subplot(grid[1, 2])
sns.violinplot(y=bmi_data, ax=ax6)
ax6.set_ylabel('BMI')
ax6.tick_params(axis='both', rotation=0)

# 调整布局并保存
plt.tight_layout()
plt.savefig('combined_plots.svg')

plt.close()
