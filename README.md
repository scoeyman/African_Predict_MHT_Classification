African Predict Masked Hypertension Model
--------------------------------------------
This project builds and evaluates machine learning models to predict masked hypertension using clinical and biometric features. The pipeline includes data preprocessing, feature imputation, model training, evaluation, and interpretation.

Requirements
--------------
- Python version: `3.12`
- pip install -r requirements.txt
- Place the raw dataset in a designated folder and update file paths in each script accordingly.

Steps
-------
1. Clean & Impute Numerical Features: `1_data_cleaning_and_num_imput.py`
     - This script reads your raw dataset and outputs a cleaned version: cleaned_data_with_imputation_papers.xlsx
     - Update `file_path` and `file_name` in the script to match your dataset location.
     - Run the script to:
       - Combine bilateral/redundant measures
       - Remove unnecessary columns
       - Impute missing numerical values

2. Impute Categorical Features: `2_cat_imputing.py`
     - This script takes the cleaned dataset from Step 1 and outputs: cleaned_data_with_imputation_papers_filled.xlsx
     - Update `file_path` in the script.
     - Run the script to generate:
       - Sheet 1: No categorical imputation
       - Sheet 2: Imputation with mode
       - Sheet 3: Imputation with median
       - Sheet 4: Imputation with mean

3. Train Models: `3_models_full.py`
     - This script trains multiple models to predict the label: abpm_overall_ht
     - Update `file_path` and `file_name` for the `cleaned_data_with_imputation_papers_filled.xlsx`.
     - By default, it uses the mode-imputed sheet — you can change this via the `names` variable.
     - Output:
       - Excel files with model scores
       - Feature selection results



## 🏆 Step 4: Finalize Best Model

**Script**: `4_model_justLASSO_XGB.py`

1. Open the model scores file and identify the model with the **highest Test ROC AUC**.
2. Edit `4_model_justLASSO_XGB.py` to use the best:
   - **Feature selection method** (e.g., LASSO)
   - **Classifier** (e.g., XGB)
3. Replace any references to `lasso_features` or `lasso_best_xgb_model` with your selected method/model.
4. Update `file_path`, `file_name`, and `feat_to_names` (for PDP plots).
5. This script will:
   - Retrain the best model
   - Perform **threshold tuning**
   - Compare against a **binary classifier**
   - Run **SHAP** and **PDP** analyses

---

## ⚖️ Step 5 (Optional): Average Binary Model Performance

**Script**: `5_binary_model_averaging.py`

1. Edit the `seeds` and `split_ratios` if needed.
2. Update `file_path`, `file_name`, and `sheet_name`.
3. Run the script to evaluate the **binary model** across multiple splits/seeds.
4. Output:
   ```
   binaryPerformance_results.xlsx
   ```

---

## 📁 Directory Structure (Suggested)

```
├── data/
│   └── raw_dataset.xlsx
├── notebooks/
├── scripts/
│   ├── 1_data_cleaning_and_num_imput.py
│   ├── 2_cat_imputing.py
│   ├── 3_models_full.py
│   ├── 4_model_justLASSO_XGB.py
│   └── 5_binary_model_averaging.py
├── outputs/
│   └── cleaned_data_with_imputation_papers.xlsx
│   └── model_scores.xlsx
│   └── binaryPerformance_results.xlsx
├── requirements.txt
└── README.md
