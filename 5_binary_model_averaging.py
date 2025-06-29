# Import Libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,StackingClassifier,  GradientBoostingRegressor, GradientBoostingClassifier
from BorutaShap import BorutaShap
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve
import shap
from tqdm import tqdm
from sklearn.inspection import PartialDependenceDisplay
import os

# Set User Defined Parameters
scoring_metric = 'roc_auc'
label = 'abpm_overall_ht'

# Set initial parameters
seeds = [1, 7, 11, 17, 21, 42, 107, 210, 1007, 2025, 11795]  # Add more seeds as needed
split_ratios = [0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.8, 0.9, 0.99]  # Add more split ratios as needed

# Load Data 
file_path = 'C:/Users/scoeyman/Desktop/af predict manual cat imputing/last time (papers)/'
file_name = 'cleaned_data_with_imputation_papers_filled.xlsx'
excel_file = pd.ExcelFile(file_path + file_name)

sheet_name = 'mode'
df = pd.read_excel(file_path + file_name, sheet_name=sheet_name)
print('processing sheet: \n', sheet_name)
dataset = df.copy()
dataset.drop('s_pp_number', axis=1, inplace=True)

categoric_features = ['sex','ethnicity','s_meds_a','s_meds_b','s_meds_d','s_meds_g','s_meds_h','s_meds_j','s_meds_m',
                      's_meds_n','s_meds_r','s_meds_s','s_meds_v','s_meds_c', 'ses_skill', 'ses_education', 'ses_income', 'ses_score', 'ses_class',
                      'prehypertensive','obese','lvh','sedentary','smoker','excessive_alcohol','dyslipidemic', 'bp_grade']

# Ensure no duplicate columns in categoric_features
categoric_features = list(dict.fromkeys(categoric_features))
categorical_data = dataset[categoric_features]
print('categoric features: \n', categoric_features)

# Assuming 'abpm_overall_ht' is the label
label_series = dataset['abpm_overall_ht']

# Convert identified columns to numeric (coerce errors to NaN for non-convertible values)
numeric_data = dataset.select_dtypes(include=[np.number])
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
numeric_data = numeric_data.drop('abpm_overall_ht', axis=1)

# Extract numeric feature names
numeric_features = numeric_data.columns.tolist()

# Check if any of the categorical features are also in numeric data
common_features = set(categoric_features) & set(numeric_features)
if common_features:
    print('\nWarning: The following features are present in both categorical and numeric lists:\n')
    print(common_features)

# Removing the common features from numeric features
numeric_features = [feature for feature in numeric_features if feature not in common_features]
print('\nnumeric features: \n', numeric_features)

# Split into Features and Labels
X = dataset.drop('abpm_overall_ht', axis=1)
y = dataset['abpm_overall_ht']

def binary_performances(y_true, y_prob, thresh, labels=['Positives','Negatives'], title=''):
    # Ensure y_prob is a 1D array of probabilities
    if y_prob.ndim > 1:
        if y_prob.shape[1] > 2:
            raise ValueError('A binary class problem is required')
        y_prob = y_prob[:, 1]

    plt.figure(figsize=[15, 12])

    # 1 -- Confusion matrix
    cm = confusion_matrix(y_true, (y_prob > thresh).astype(int))
    plt.subplot(221)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                      annot_kws={"size": 14}, fmt='g')
    cmlabels = ['True Negatives', 'False Positives',
                'False Negatives', 'True Positives']
    for i, t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])
    plt.title('Confusion Matrix', size=15)
    plt.xlabel('Predicted Values', size=13)
    plt.ylabel('True Values', size=13)

    #2 -- Distributions of Predicted Probabilities of both classes
    plt.subplot(222)
    plt.hist(y_prob[y_true==1], density=True, bins=25,
              alpha=.5, color='green',  label=labels[0])
    plt.hist(y_prob[y_true==0], density=True, bins=25,
              alpha=.5, color='red', label=labels[1])
    plt.axvline(thresh, color='blue', linestyle='--', label='Decision Threshold')
    plt.xlim([0,1])
    plt.title('Distributions of Predictions', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")

    #3 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(223)
    plt.plot(fp_rates, tp_rates, color='orange',
              lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve', size=15)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=.3)

    #4 -- PR curve with annotated decision point
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    plt.subplot(224)
    plt.plot(recalls, precisions, color='orange',
              lw=1, label='PR curve (area = %0.3f)' % pr_auc)
    no_skill_score = len(y_true[y_true==1]) / len(y_true)
    plt.plot([0, 1], [no_skill_score, no_skill_score], lw=1, linestyle='--', color='grey')
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(tp/(tp+fn), tp/(tp+fp), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', size=13)
    plt.ylabel('Precision', size=13)
    plt.title('Precision-Recall Curve', size=15)
    plt.legend(loc="upper right")
    plt.subplots_adjust(wspace=.3)
    plt.suptitle(title)
    # plt.show(block=True)

    tn, fp, fn, tp = [i for i in cm.ravel()]
    accuracy = (tp+tn)/(tp+fp+tn+fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2*(precision * recall) / (precision + recall)

    results = {
        'Decision Threshold': thresh,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Accuracy": accuracy, "Precision": precision,
        "Recall": recall, "F1 Score": F1,
        "AUC": roc_auc, "PR AUC":pr_auc
    }

    prints = [f"{kpi}: {round(score, 3)}" for kpi,score in results.items()]
    prints = ' | '.join(prints)
    print(prints)
    return results

# Initialize DataFrame for results
results = []

# Iterate over seeds and split ratios
for seed in seeds:
    for split_ratio in split_ratios:
        X_train = []
        X_test = []
        y_train = []
        y_test = []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=seed, stratify=y
        )

        # No Model Evaluation
        y_no_model = X_test['sbp'].apply(lambda x: 1 if x > 120 else 0)
        results_05 = binary_performances(y_test, y_no_model, thresh=0.5, title=f'No Model Evaluation @ Seed {seed}, Split {split_ratio}')

        # Append results
        results.append({
            'Seed': seed,
            'Split Ratio': split_ratio,
            **results_05  # Unpack dictionary
        })

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_file = os.path.join(file_path, "binaryPerformance_results.xlsx")
results_df.to_excel(results_file, index=False)
print(f"Results saved to {results_file}")