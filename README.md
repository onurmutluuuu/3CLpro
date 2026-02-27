# 🧬 Machine Learning Approaches to Predict SARS-CoV-2 3CLpro Inhibitors

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![ML Framework](https://img.shields.io/badge/Scikit_Learn-v1.6.1-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Academic_Paper-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)
[![DOI](https://img.shields.io/badge/DOI-Pending-b31b1b.svg)](https://doi.org/10.xxxx/xxxxx)

> **Official Repository** for the research paper: *"Machine learning approaches to predict SARS-CoV-2 3CLpro inhibitors: A comparative study of in-silico and in-vitro profiles of natural compounds"*

---

## 📖 Abstract

This study presents a comprehensive machine learning framework to identify potent natural inhibitors against **SARS-CoV-2 3CLpro** (Main Protease). By integrating **in-silico binding affinity predictions** with **in-vitro enzymatic inhibition data**, we developed a multi-stage predictive pipeline.

The dataset consists of **384 natural compounds**, characterized by **23 molecular descriptors** (PaDEL). The workflow operates in three distinct phases, transitioning from regression to hybrid classification tasks.

---

## ⚙️ Methodology & Workflow

Our pipeline consists of three sequential steps, designed to maximize predictive accuracy for both binding affinity and inhibition efficacy.

### 🔹 **Step 1: Binding Affinity Prediction (Regression)**
* **Objective:** Predict the binding affinity (kcal/mol) obtained from molecular docking.
* **Best Model:** **XGBoost** ($R^2 = 0.85$)
* **Key Insight:** Regression models effectively captured the quantitative structure-activity relationship.

### 🔹 **Step 2: Inhibition Prediction (Threshold 50%)**
* **Objective:** Classify compounds as Active/Inactive based on a **50% inhibition threshold**.
* **Input:** Molecular descriptors only.
* **Best Model:** **CatBoost** (Accuracy = 0.81).

### 🔹 **Step 3: Hybrid Classification (Threshold 20%)**
* **Objective:** Detect compounds with at least **20% inhibition**.
* **Unique Feature:** Uses **In-Silico Binding Affinity** as an input feature along with molecular descriptors.
* **Best Model:** **MLP (Multi-Layer Perceptron)** (AUC = 0.76)
* **Key Insight:** Incorporating docking scores significantly improves the detection of lower-efficacy compounds.

---

## 📊 Performance Highlights

| Metric | Step 1 (Regression) | Step 2 (Class - 50%) | Step 3 (Class - 20%) |
| :--- | :---: | :---: | :---: |
| **Target** | Binding Affinity | Inhibition > 50% | Inhibition > 20% |
| **Algorithm** | **XGBoost** | **CatBoost** | **MLP** |
| **Performance** | **$R^2$: 0.85** | **Accuracy: 81%** | **AUC: 0.76** |
| **Input** | 23 Features | 23 Features | 23 Features + **Binding Affinity** |

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/3CLpro_ML_Project.git

cd 3CLpro_ML_Project

