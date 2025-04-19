<p align="center">
  <a href="https://colab.research.google.com/github/your-username/customerId-embedding-analysis/blob/main/notebooks/customerId_feature_analysis_fancy_intro.ipynb">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-red">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-brightgreen">
</p>

**Understanding why using raw customer identifiers (`customerId`) in machine learning models can cause overfitting, and how to address it with better design and regularization.**

---

## 📚 Contents

- **Deep dive** into why `customerId` is dangerous when misused
- **PyTorch experiments** showing overfitting and generalization
- **Visualization** of loss curves and AUC scores
- **Rescue model** using Dropout and L2 regularization
- **Mathematical proof** showing variance \\( \\propto \\frac{1}{n} \\)

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/saibaba/customerId-embedding-analysis.git
cd customerId-embedding-analysis

# Install dependencies
pip install -r requirements.txt
