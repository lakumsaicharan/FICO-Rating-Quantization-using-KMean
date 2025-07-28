# FICO-Rating-Quantization-using-KMean

This project implements a credit rating system by bucketing borrower FICO scores using a machine learning approach. The final buckets are used to assign a **rating**, which can be used as input to downstream default prediction models.

## 🔍 Objective
Transform numerical `fico_score` values into categorical credit **ratings** using quantization techniques — optimized to reduce prediction error in risk models.

## ✅ Method Used
- **MSE-based Quantization** using **1D KMeans Clustering**
- 5 discrete buckets representing credit quality:
  - `1 = Excellent`
  - `5 = Poor`

## 📁 Files Included
| File | Description |
|------|-------------|
| `fico_rating_model.ipynb` | Main script to process the data and assign ratings |
| `Loan_Data.csv` | Sample data (10,000 records) |
| `fico_ratings_output.csv` | Final dataset with ratings |
| `plots/` | Charts showing distribution and default trends |

## 🧠 Tools Used
- Python (Pandas, NumPy)
- scikit-learn (KMeans)
- seaborn / matplotlib
