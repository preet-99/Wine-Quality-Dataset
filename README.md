
# 🍷 Wine Quality Prediction using Random Forest

A machine learning pipeline to predict wine quality using the Wine Quality dataset. Includes preprocessing, feature scaling, encoding, and training a RandomForest model with evaluation.



## 📂 Project Structure

```text
├── WineQT.csv        # Dataset file (input data)  
├── model.pkl         # Trained Random Forest model (saved after training)  
├── pipeline.pkl      # Preprocessing pipeline (saved after training)  
├── input.csv         # Test set used for inference  
├── output.csv        # Predictions with target values  
├── main.py           # Main Python script (entry point)  
├── requirements.txt  # Project dependencies  
└── README.md         # Project documentation 
````

## ⚙️ Features
```text
● Preprocessing pipeline:
  ● Imputation of missing values
  ● Standard scaling for numerical features
  ● One-hot encoding for categorical features (if any)
  ● Train/test split using StratifiedShuffleSplit
● Random Forest model (Regressor or Classifier)
● Model persistence using joblib
● Evaluation:
  ● Regression: MSE, RMSE, MAE, R² score
  ● Classification: Accuracy, Confusion Matrix, Classification Report
● Feature importance visualization
```    
## Authors

-  @preet-99 - Preet Vishwakarma

## 🛠️ Installation
```bash
 git clone https://github.com/preet-99/Wine-Quality-Dataset.git
cd wine-quality-prediction
pip install -r requirements.txt

```