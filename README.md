<h1 align="center">
MF-NIPT: Multi-Factor Statistical Modeling for Non-Invasive Prenatal Testing Time Selection and Fetal Abnormality Prediction
</h1>
<p align="center">
<img src="https://img.shields.io/badge/OS-Ubuntu22.4-blue" />
<img src="https://img.shields.io/badge/Python-3.13-red" />
<img src="https://img.shields.io/badge/Build-Success-green" />
<img src="https://img.shields.io/badge/License-BSD-blue" />
<img src="https://img.shields.io/badge/Release-0.1-blue" />
</p>

Non-Invasive Prenatal Testing (NIPT), which analyzes cell-free fetal DNA (cffDNA) in maternal plasma, has emerged as a reliable method for detecting fetal chromosomal abnormalities. However, conventional NIPT protocols often neglect inter-individual maternal variations—such as body mass index (BMI), gestational age, and other physiological parameters—that significantly influence fetal fraction (FF) concentration and assay accuracy. To address this limitation, we propose a multi-factor statistical modeling framework, MF-NIPT. The framework synergistically integrates statistical inference and machine learning to enable personalized selection of the optimal NIPT timing and accurate prediction of fetal anomalies. The MF-NIPT framework comprises two core modules: Multi-Factor Relationship Modeling (MFRM) and Multi-Factor Abnormality Prediction (MFAP). Experimental results demonstrate that MF-NIPT accurately predicts FF, identifies the earliest viable gestational week for NIPT, and achieves an F1-score of 0.918 in abnormality detection for female fetuses. This work establishes a mathematically interpretable and clinically actionable foundation for individualized, risk-aware NIPT decision-making.


<p align="center">
<img src="./doc/img/fig.1.jpg" width=100%/> <br>
<b>Figure 1.</b> Structure of MF-NIPT.  
</p>



## Conda Environment Setup

``` shell
conda create -n mfnipt python=3.13 -y
conda activate mfnipt
```

## Dataset

### **CUMCM-2025C**

The data in this study originate from cffDNA sequencing results of pregnant women, categorized by fetal sex. The dataset used is CUMCM-2025C, provided by the China Undergraduate Mathematical Contest in Modeling (CUMCM). It contains 1,687 maternal samples, including 1,082 male-fetus samples and 605 female-fetus samples. Maternal feature variables include gestational week (t), body mass index (BMI), height (H), weight (W), age (A), and GC content (GC). Fig. 2 shows the distribution of key factors in the CUMCM-2025C dataset: Z-scores of chromosome X, concentration of chromosome Y, and maternal BMI.

<p align="center">
<img src="./doc/img/fig.dataset.distr.jpg" width=100%/> <br>
<b>Figure 2.</b> Distribution of CUMCM-2025C data, including Z-score of chromosome X, concentration of chromosome Y, and maternal BMI—three key factors.  
</p>



### Load Dataset

```python
import pandas as pd
from datasets import load_dataset

# load data from huggingface hub
repo_id = "cumcm-dataset/CUMCM-2025c-dataset"
dataset = load_dataset(repo_id)
data = dataset['train'].to_pandas()  # to pandas DataFrame

# data filter
female_fetus_data = data[data[' Fetal Type (e.g., Male Fetus)'] == 'Female Fetus']
male_fetus_data = data[data[' Fetal Type (e.g., Male Fetus)'] == 'Male Fetus']

x_chromosome_z_score_female = female_fetus_data[' Z-Score of Chromosome X']
y_chromosome_concentration_male = male_fetus_data['Concentration of Chromosome Y']
bmi_data = data[" Pregnant Woman's BMI (BMI: Body Mass Index)"]
```

## Train

``` python
check_dir(base_path+"output/csv/")
check_dir(base_path+"output/log/")
check_dir(base_path+"output/pt/")
check_dir(base_path+"output/board/")

```

output:

``` shell
ddd
```

## Test

``` python
check_dir(base_path+"output/csv/")
check_dir(base_path+"output/log/")
check_dir(base_path+"output/pt/")
check_dir(base_path+"output/board/")

```

output:

``` shell

```
