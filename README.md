FinanKu Credit Card Delay Prediction

Overview
This project predicts credit card payment delays for FinanKu customers using engineered behavioral features and machine learning models.

Features
- Data cleaning and preprocessing
- Custom feature engineering
- Predictive modeling with Logistic Regression, Random Forest, and XGBoost
- Model evaluation and improvement strategies

Dataset
Project uses anonymized FinanKu customer data with variables like branch, age, income, balance, and payment outcome.

Getting Started
1. Clone repository:
   ```
   git clone https://github.com/[username]/[repo-name].git
   cd [repo-name]
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the notebook:
   Open `PredictionModel.ipynb` in Jupyter and execute all cells.

File Structure

| File                    | Description                           |
|-------------------------|---------------------------------------|
| PredictionModel.ipynb   | Notebook for data processing and modeling |
| FinanKu-Data-All.csv    | Main dataset                          |
| FinanKu-Data-Validasi.csv| Validation/test dataset              |
| README.md               | Documentation                         |

Results
- All models evaluated with recall as main metric
- Best Recall Score: >40% (further improvement discussed)
- Practical recommendations for handling imbalance and increasing model performance given

Next Steps / Improvements
- Add more data or use advanced sampling (SMOTE)
- Try other algorithms (LightGBM, CatBoost)
- Tune hyperparameters further and test additional features


License
For research use only. Contact author about data usage.

Acknowledgments
Thanks to FinanKu for dataset and business context.


