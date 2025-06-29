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
import inspect

# Set User Defined Parameters
SEED = 11
cv_metric = StratifiedKFold(5, random_state=SEED, shuffle=True)
scoring_metric = 'roc_auc'
split_ratio = 0.2
label = 'abpm_overall_ht'

# Load Data 
file_path = 'C:/Users/scoeyman/Desktop/af predict manual cat imputing/last time (papers)/'
file_name = 'cleaned_data_with_imputation_filled.xlsx'
excel_file = pd.ExcelFile(file_path + file_name)

# Dictionary to store scores
scores = {
    'Model': [],
    'Train Score ROC AUC': [],
    'Train Score Stdv': [],
    'Test Score ROC AUC': [],
    'Test Score Stdv': []
}

# Function to add scores to the dictionary
def add_scores(model_name, train_scores, test_scores):
    scores['Model'].append(model_name)
    scores['Train Score ROC AUC'].append(f"{train_scores.mean():.6f}")
    scores['Train Score Stdv'].append(f"{train_scores.std():.6f}")
    scores['Test Score ROC AUC'].append(f"{test_scores.mean():.6f}")
    scores['Test Score Stdv'].append(f"{test_scores.std():.6f}")

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
    print('Warning: The following features are present in both categorical and numeric lists:')
    print(common_features)

# Removing the common features from numeric features
numeric_features = [feature for feature in numeric_features if feature not in common_features]
print('numeric features: \n', numeric_features)

# Split into Features and Labels
X = dataset.drop('abpm_overall_ht', axis=1)
y = dataset['abpm_overall_ht']

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state = SEED, stratify = y)

### Preprocess Features
print('Preprocessing Features')
X_train_numeric = X_train[numeric_features]
X_train_categoric = X_train[categoric_features]
X_test_numeric = X_test[numeric_features]
X_test_categoric = X_test[categoric_features]

# Scale Numerical Features
print('Performing MinMax Scaling')
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
scaler = MinMaxScaler()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train_scaled[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test_scaled[numeric_features])

################## Select Features using LASSO ##################
print('Performing Feature Selection with LASSO')
selector = LassoCV()
selector.fit(X_train_scaled,y_train)
coefficients = selector.coef_
importance = np.abs(coefficients)
lasso_features = np.array(X_train_scaled.columns)[importance > 0]
print('Best LASSO features: ', lasso_features)

X_train_final = X_train_scaled.copy()
X_test_final = X_test_scaled.copy()
y_train_final = y_train.copy()
y_test_final = y_test.copy()

X_train_final = X_train_final[lasso_features]
X_test_final = X_test_final[lasso_features]
y_train_final = y_train_final
y_test_final = y_test_final

# Gradient boost
xgb_model = XGBClassifier(random_state=SEED)
params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 5, 10],
        'subsample': [0.5, 1.0, 'uniform'],
        'gamma':[0, 5.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
        }
xgb_search = BayesSearchCV(xgb_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
xgb_search.fit(X_train_final, y_train_final)
lasso_best_xgb_model = xgb_search.best_estimator_
xgb_model_scores_train = cross_val_score(lasso_best_xgb_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
msg = "TRAINING SET - LASSO Tunned XGB Model Score: %f (%f)" % (xgb_model_scores_train.mean(), xgb_model_scores_train.std())
print(msg)
xgb_model_scores_test = cross_val_score(lasso_best_xgb_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
msg = "TEST SET - LASSO Tunned XGB Model Score: %f (%f)" % (xgb_model_scores_test.mean(), xgb_model_scores_test.std())
print(msg)
add_scores('LASSO XGB', xgb_model_scores_train, xgb_model_scores_test)




##### Perform Threshold Tunning ###### - finding optimal decision threshold
y_probs_tune = lasso_best_xgb_model.predict_proba(X_train_scaled[lasso_features].values)
y_probs_tune = y_probs_tune[:,1]
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
thresholds = np.arange(0.01,0.99,0.001)
thresholds_df = pd.DataFrame({'threshold':thresholds})
i = 1
for _, val_index in cv.split(X_train_scaled[lasso_features],y_train):
    X_val_cv = X_train_scaled[lasso_features].iloc[val_index]
    y_val_cv = y_train_final.iloc[val_index]
    y_probs_tune = lasso_best_xgb_model.predict_proba(X_val_cv.values)
    y_probs_tune = y_probs_tune[:,1]
    fscores = []
    for threshold in thresholds:
        y_pred = y_probs_tune > threshold
        fscore = roc_auc_score(y_val_cv, y_pred)
        fscores.append(fscore)
    thresholds_df = pd.concat([thresholds_df,pd.DataFrame({'Fold'+str(i):fscores})],axis=1)
    i = i + 1
average_fscores = pd.DataFrame({'average':thresholds_df.iloc[:,1:10+1].mean(axis=1)})
pr_ix = np.nanargmax(average_fscores)
decision_threshold = thresholds[pr_ix]
print('Optimal decision threshold: '+ str(decision_threshold))


##### Binary Peformance Function and Plotting ##### 
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
    plt.show(block=True)

    tn, fp, fn, tp = [i for i in cm.ravel()]
    accuracy = (tp+tn)/(tp+fp+tn+fn)
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

y_no_model = X_test['sbp'].apply(lambda x: 1 if x > 120 else 0)
no_model_test_results_05 = binary_performances(y_test, y_no_model, thresh = 0.5, title = 'No Model Evaluation: Testing Set @ 0.5')

# Ensure that the feature names are aligned
feature_names = X_test_scaled[lasso_features].columns
y_probs = lasso_best_xgb_model.predict_proba(X_test_scaled[lasso_features])
y_probs = y_probs[:,1]
lassoXGB_test_results = binary_performances(y_test, y_probs, thresh = decision_threshold, title = 'Classifier Model Evaluation: Testing Set')

# Map of feature abbreviations to descriptive names -> WILL NEED TO BE UPDATED BASED OFF FEATURES SELECTED
feat_to_names = {
    'ses_skill': 'Socio-Economic Status – Skill Level',
    'ses_income': 'Socio-Economic Status – Total Household Income',
    'ses_score': 'Socio-Economic Status – Score',
    'ses_class': 'Socio-Economic Status – Class',
    'ses_education': 'Socio-Economic Status – Education',
    'bw': 'Body Weight',
    'bp_grade': 'Clinic Blood Pressure Grade',
    'travel_minutes_week': 'Minutes/Week Traveled Using Bicycle or Walking',
    'lvpwd': 'LV Posterior Wall Thickness at Diastole',
    'lvpws': 'LV Posterior Wall Thickness at Systole',
    'lvs_mass': 'LV Mass at Systole',
    'lad': 'Left Atrium Diameter',
    'aod': 'Aortic Root Diameter',
    'mv_ea_ratio': 'Mitral Valve E to A Ratio',
    'ras_renin': 'Ang I + Ang II',
    'dhea_s': 'Dehydroepiandrosterone Sulfate',
    'cp': 'C-peptide',
    'glu_u_sz_creat': 'Glucose in mg/gcreat',
    'uric_u_sz_mgdl': 'Uric Acid in mg/dL',
    'phos_u_sz_mgdl': 'Phosphorus in mg/dL',
    'cr_u_sz': 'Chromium',
    'sbp': 'Systolic Blood Pressure',
    'cpp': 'Central Aortic Pulse Pressure',
    'map_syst': 'Mean Arterial Pressure Systole',
    'sex': 'Participant Gender',
    'v': 'Participant Ethnicity',
    's_meds_a': 'Medication for Alimentary Tract and Metabolism',
    'prehypertensive': 'Prehypertensive',
    'sedentary': 'Sedentary',
    'smoker': 'Smoker',
    'excessive_alcohol': 'Alcohol Consumption',
    'dyslipidemic': 'Dyslipidemic',
    'obese': 'Patient Obesity',
    'age_b': 'Participant Age',
    'ethnicity': 'Patient Ethnicity'
}

# Map the feature names for display
mapped_feature_names = [feat_to_names.get(name, name) for name in feature_names]

# Manual exclusions
manual_exclusions = ['']
# manual_exclusions = ['ses_education', 'ses_income','obese', 'age_b', 'sex', 'ethnicity', 's_meds_a', 'excessive_alcohol', 'dyslipidemic'] ## REMOVED FOR PAPER IAMGE -> NO EFFECT ON MODEL AFTER REVEWING PDPS

# Use the full set of features (lasso_features) for the SHAP explainer
explainer = shap.Explainer(lasso_best_xgb_model.predict, X_test_scaled[lasso_features].values)

# Compute SHAP values using all features
shap_values = explainer(X_test_scaled[lasso_features])
# Ensure lasso_features is a list
lasso_features = list(lasso_features)
# Filter SHAP values to exclude unwanted features
filtered_indices = [lasso_features.index(f) for f in lasso_features if f not in manual_exclusions]
shap_values_filtered = shap.Explanation(
    values=shap_values.values[:, filtered_indices],
    base_values=shap_values.base_values,
    data=shap_values.data[:, filtered_indices],
    feature_names=[mapped_feature_names[i] for i in filtered_indices]
)

# SHAP Beeswarm Plot
shap.plots.beeswarm(shap_values_filtered, max_display=len(filtered_indices))

# Calculate the mean absolute SHAP values for filtered features
feature_importance = np.abs(shap_values_filtered.values).mean(axis=0)

# Create a DataFrame to store the feature importance scores
feature_importance_df = pd.DataFrame({
    'Feature': [mapped_feature_names[i] for i in filtered_indices],
    'Importance': feature_importance
})

# Sort the DataFrame by importance scores in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Features')
plt.title('Feature Importance (Mean Absolute SHAP Values)')
plt.gca().invert_yaxis()  # Invert y-axis to display the most important features at the top
plt.show()

##### Partial Dependence Plots for top features (filtered features) #####
n_features = len(filtered_indices)  # Use filtered indices length for number of features
n_cols = 4  # Adjust the number of columns as needed
n_rows = (n_features + n_cols - 1) // n_cols  # Calculate the required number of rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Loop through filtered features and plot PDPs
for i, idx in enumerate(filtered_indices):
    feature_name = lasso_features[idx]  # Get the feature name from the filtered indices
    display_name = mapped_feature_names[idx]  # Get the mapped feature name for display
    
    print(f"Plotting PDP for feature: {display_name}")

    # Plot the PDP for the i-th feature
    PartialDependenceDisplay.from_estimator(
        estimator=lasso_best_xgb_model,
        X=X_test_scaled[lasso_features],
        features=[feature_name],
        ax=axes[i]
    )
    axes[i].set_title(f'{display_name}', fontsize=10)

# Hide any unused subplots (if the number of features is not divisible by n_cols)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()
