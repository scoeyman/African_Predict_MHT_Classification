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
import sys

# Set  Defined Parameters
SEED = 11
cv_metric = StratifiedKFold(5, random_state=SEED, shuffle=True)
scoring_metric = 'roc_auc'                      # accuracy, precison, etc for future note
split_ratio = 0.2
label = 'abpm_overall_ht'

# Load Data 
file_path = 'C:/Users/scoeyman/Desktop/af predict manual cat imputing/last time (papers)/'
file_name = 'cleaned_data_with_imputation_filled.xlsx'
excel_file = pd.ExcelFile(file_path + file_name)

# Dictionary to store scores from model training
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

# Loop through the sheets - just mode is shown, change to mean or median if needed
names = ['mode']
for sheet_name in names:
    # Loading excel sheet 
    df = pd.read_excel(file_path + file_name, sheet_name=sheet_name)
    print('processing sheet: \n', sheet_name)
    dataset = df.copy()

    # Dropping patient number identifier
    dataset.drop('s_pp_number', axis=1, inplace=True)
    
    # Defining categoric features
    categoric_features = ['sex','ethnicity','s_meds_a','s_meds_b','s_meds_d','s_meds_g','s_meds_h','s_meds_j','s_meds_m',
                          's_meds_n','s_meds_r','s_meds_s','s_meds_v','s_meds_c', 'ses_skill', 'ses_education', 'ses_income', 
                          'ses_score', 'ses_class','prehypertensive','obese','lvh','sedentary','smoker','excessive_alcohol',
                          'dyslipidemic', 'bp_grade']     # slightly different than code "1_data_cleaning..." due to features being dropped
    
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

    # Preprocess Features
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


    ################## Performing No Feature Selection  ##################
    X_train_final = X_train_scaled.copy()
    X_test_final = X_test_scaled.copy()
    y_train_final = y_train.copy()
    y_test_final = y_test.copy()

    # Logisitc regression
    lr_model = LogisticRegression(random_state=SEED)
    params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear','saga'],
            'class_weight':[None,'balanced']
        }
    lr_search = BayesSearchCV(lr_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    lr_search.fit(X_train_final, y_train_final)
    no_fs_best_lr_model = lr_search.best_estimator_
    lr_model_scores_train = cross_val_score(no_fs_best_lr_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Base LR Model Score: %f (%f)" % (lr_model_scores_train.mean(), lr_model_scores_train.std())
    print(msg)
    lr_model_scores_test = cross_val_score(no_fs_best_lr_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Base LR Model Score: %f (%f)" % (lr_model_scores_test.mean(), lr_model_scores_test.std())
    print(msg)
    add_scores('Base LR', lr_model_scores_train, lr_model_scores_test)
    
    # Random forest
    rf_model = RandomForestClassifier(random_state=SEED)
    params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None ,'sqrt','log2'],
            'criterion': ['gini', 'entropy'],
            'class_weight':[None,'balanced'],
            }
    rf_search = BayesSearchCV(rf_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    rf_search.fit(X_train_final, y_train_final)
    no_fs_best_rf_model = rf_search.best_estimator_
    rf_model_scores_train = cross_val_score(no_fs_best_rf_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Base RF Model Score: %f (%f)" % (rf_model_scores_train.mean(), rf_model_scores_train.std())
    print(msg)
    rf_model_scores_test = cross_val_score(no_fs_best_rf_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Base RF Model Score: %f (%f)" % (rf_model_scores_test.mean(), rf_model_scores_test.std())
    print(msg)
    add_scores('Base RF', rf_model_scores_train, rf_model_scores_test)

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
    no_fs_best_xgb_model = xgb_search.best_estimator_
    xgb_model_scores_train = cross_val_score(no_fs_best_xgb_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Base XGB Model Score: %f (%f)" % (xgb_model_scores_train.mean(), xgb_model_scores_train.std())
    print(msg)
    xgb_model_scores_test = cross_val_score(no_fs_best_xgb_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Base XGB Model Score: %f (%f)" % (xgb_model_scores_test.mean(), xgb_model_scores_test.std())
    print(msg)
    add_scores('Base XGB', xgb_model_scores_train, xgb_model_scores_test)

    # ANN / MLP
    mlpc_model = MLPClassifier(random_state=SEED,max_iter=2000)
    params = {
            'activation':['relu','tanh'],
            'solver':['sgd','adam'],
            'learning_rate':['constant','adaptive'],
            'alpha':[0.0001, 0.001, 0.01]
            }
    mlpc_search = BayesSearchCV(mlpc_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    mlpc_search.fit(X_train_final, y_train_final)
    no_fs_best_mlpc_model = mlpc_search.best_estimator_
    mlpc_model_scores_train = cross_val_score(no_fs_best_mlpc_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Base ANN Model Score: %f (%f)" % (mlpc_model_scores_train.mean(), mlpc_model_scores_train.std())
    print(msg)
    mlpc_model_scores_test = cross_val_score(no_fs_best_mlpc_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Base ANN Model Score: %f (%f)" % (mlpc_model_scores_test.mean(), mlpc_model_scores_test.std())
    print(msg)
    add_scores('Base ANN (MLP)', xgb_model_scores_train, xgb_model_scores_test)

    # Stacking
    base_learners = [
            ('LR', no_fs_best_lr_model),
            ('RF', no_fs_best_rf_model),
            ('ANN', no_fs_best_mlpc_model),
            ('XGB', no_fs_best_xgb_model),
            ]
    no_fs_stk_model = StackingClassifier(estimators=base_learners, final_estimator = lr_model, cv = cv_metric)
    no_fs_stk_model.fit(X_train_final, y_train_final)
    stk_model_scores_train = cross_val_score(no_fs_stk_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Base Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_train.mean(), stk_model_scores_train.std())
    print(msg)
    stk_model_scores_test = cross_val_score(no_fs_stk_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Base Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_test.mean(), stk_model_scores_test.std())
    print(msg)
    add_scores('Base STACK', stk_model_scores_train, stk_model_scores_test)


    ################## Perform Manual Feature Selection ##################
    manually_selected_features = ['sex','ethnicity','age_b','sbp','dbp',
                         'map','pp','hr','glucose_rerun','trig','ldl','hdl','chol','cotinine',
                         'ggt','smoker','excessive_alcohol','sedentary','prehypertensive',
                         'obese','lvm_cube','lvh','imtn_mean', 'pwv', 'mcp_1', 'crp']   # chosen from literature reviews
    
    X_train_final = X_train_scaled.copy()
    X_test_final = X_test_scaled.copy()
    y_train_final = y_train.copy()
    y_test_final = y_test.copy()

    X_train_final = X_train_final[manually_selected_features]
    X_test_final = X_test_final[manually_selected_features]
    y_train_final = y_train_final
    y_test_final = y_test_final

    # Logisitc regression
    lr_model = LogisticRegression(random_state=SEED)
    params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear','saga'],
            'class_weight':[None,'balanced']
        }
    lr_search = BayesSearchCV(lr_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    lr_search.fit(X_train_final, y_train_final)
    manual_best_lr_model = lr_search.best_estimator_
    lr_model_scores_train = cross_val_score(manual_best_lr_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Manually Tunned LR Model Score: %f (%f)" % (lr_model_scores_train.mean(), lr_model_scores_train.std())
    print(msg)
    lr_model_scores_test = cross_val_score(manual_best_lr_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Manually Tunned LR Model Score: %f (%f)" % (lr_model_scores_test.mean(), lr_model_scores_test.std())
    print(msg)
    add_scores('MS LR', lr_model_scores_train, lr_model_scores_test)

    # Random forest
    rf_model = RandomForestClassifier(random_state=SEED)
    params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None,'sqrt','log2'],
            'criterion': ['gini', 'entropy'],
            'class_weight':[None,'balanced'],
            }
    rf_search = BayesSearchCV(rf_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    rf_search.fit(X_train_final, y_train_final)
    manual_best_rf_model = rf_search.best_estimator_
    rf_model_scores_train = cross_val_score(manual_best_rf_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Manually Tunned RF Model Score: %f (%f)" % (rf_model_scores_train.mean(), rf_model_scores_train.std())
    print(msg)
    rf_model_scores_test = cross_val_score(manual_best_rf_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TESTING SET - Manually Tunned RF Model Score: %f (%f)" % (rf_model_scores_test.mean(), rf_model_scores_test.std())
    print(msg)
    add_scores('MS RF', rf_model_scores_train, rf_model_scores_test)

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
    manual_best_xgb_model = xgb_search.best_estimator_
    xgb_model_scores_train = cross_val_score(manual_best_xgb_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Manually Tunned XGB Model Score: %f (%f)" % (xgb_model_scores_train.mean(), xgb_model_scores_train.std())
    print(msg)
    xgb_model_scores_test = cross_val_score(manual_best_xgb_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Manually Tunned XGB Model Score: %f (%f)" % (xgb_model_scores_test.mean(), xgb_model_scores_test.std())
    print(msg)
    add_scores('MS XGB', xgb_model_scores_train, xgb_model_scores_test)

    # ANN / MLP
    mlpc_model = MLPClassifier(random_state=SEED,max_iter=2000)
    params = {
            #'hidden_layer_sizes': [(50,),(100,),(50,50),(100,50,25)],
            'activation':['relu','tanh'],
            'solver':['sgd','adam'],
            'learning_rate':['constant','adaptive'],
            'alpha':[0.0001, 0.001, 0.01]
            }
    mlpc_search = BayesSearchCV(mlpc_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    mlpc_search.fit(X_train_final, y_train_final)
    manual_best_mlpc_model = mlpc_search.best_estimator_
    mlpc_model_scores_train = cross_val_score(manual_best_mlpc_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Manually Tunned ANN Model Score: %f (%f)" % (mlpc_model_scores_train.mean(), mlpc_model_scores_train.std())
    print(msg)
    mlpc_model_scores_test = cross_val_score(manual_best_mlpc_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Manually Tunned ANN Model Score: %f (%f)" % (mlpc_model_scores_test.mean(), mlpc_model_scores_test.std())
    print(msg)
    add_scores('MS ANN (MLP)', mlpc_model_scores_train, mlpc_model_scores_test)

    # Stacking
    base_learners = [
            ('LR', manual_best_lr_model),
            ('RF', manual_best_rf_model),
            ('ANN', manual_best_mlpc_model),
            ('XGB', manual_best_xgb_model),
            ]
    manual_stk_model = StackingClassifier(estimators=base_learners, final_estimator = lr_model, cv = cv_metric)
    manual_stk_model.fit(X_train_final, y_train_final)
    stk_model_scores_train = cross_val_score(manual_stk_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - Manually Tunned Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_train.mean(), stk_model_scores_train.std())
    print(msg)
    stk_model_scores_test = cross_val_score(manual_stk_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - Manually Tunned Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_test.mean(), stk_model_scores_test.std())
    print(msg)
    add_scores('MS STACK', stk_model_scores_train, stk_model_scores_test)


    ################## Select Features using RFE-SVM ##################
    print('Performing Feature Selection with RFE-SVM')
    estimator = SVC(random_state=SEED, probability=True, kernel='linear')
    selector = RFECV(estimator = estimator, cv = cv_metric, scoring = scoring_metric, verbose=1)
    selector.fit(X_train_scaled,y_train)
    print(selector.support_)
    print(selector.ranking_)
    selector.support_

    rfe_svm_features = X_train_scaled.columns[selector.support_]
    print('Best RFECV-SVM features:', rfe_svm_features)

    X_train_final = X_train_scaled.copy()
    X_test_final = X_test_scaled.copy()
    y_train_final = y_train.copy()
    y_test_final = y_test.copy()

    X_train_final = X_train_final[rfe_svm_features]
    X_test_final = X_test_final[rfe_svm_features]
    y_train_final = y_train_final
    y_test_final = y_test_final

    # Logistic regression
    lr_model = LogisticRegression(random_state=SEED)
    params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear','saga'],
            'class_weight':[None,'balanced']
        }
    lr_search = BayesSearchCV(lr_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    lr_search.fit(X_train_final, y_train_final)
    rfe_svm_best_lr_model = lr_search.best_estimator_
    lr_model_scores_train = cross_val_score(rfe_svm_best_lr_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - RFECV-SVM Tunned LR Model Score: %f (%f)" % (lr_model_scores_train.mean(), lr_model_scores_train.std())
    print(msg)
    lr_model_scores_test = cross_val_score(rfe_svm_best_lr_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - RFECV-SVM Tunned LR Model Score: %f (%f)" % (lr_model_scores_test.mean(), lr_model_scores_test.std())
    print(msg)
    add_scores('RFE LR', lr_model_scores_train, lr_model_scores_test)

    # Random forest
    rf_model = RandomForestClassifier(random_state=SEED)
    params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None,'sqrt','log2'],
            'criterion': ['gini', 'entropy'],
            'class_weight':[None,'balanced'],
            }
    rf_search = BayesSearchCV(rf_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    rf_search.fit(X_train_final, y_train_final)
    rfe_svm_best_rf_model = rf_search.best_estimator_
    rf_model_scores_train = cross_val_score(rfe_svm_best_rf_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - RFECV-SVM Tunned RF Model Score: %f (%f)" % (rf_model_scores_train.mean(), rf_model_scores_train.std())
    print(msg)
    rf_model_scores_test = cross_val_score(rfe_svm_best_rf_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - RFECV-SVM Tunned RF Model Score: %f (%f)" % (rf_model_scores_test.mean(), rf_model_scores_test.std())
    print(msg)
    add_scores('RFE RF', rf_model_scores_train, rf_model_scores_test)

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
    rfe_svm_best_xgb_model = xgb_search.best_estimator_
    xgb_model_scores_train = cross_val_score(rfe_svm_best_xgb_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - RFECV-SVM Tunned XGB Model Score: %f (%f)" % (xgb_model_scores_train.mean(), xgb_model_scores_train.std())
    print(msg)
    xgb_model_scores_test = cross_val_score(rfe_svm_best_xgb_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - RFECV-SVM Tunned XGB Model Score: %f (%f)" % (xgb_model_scores_test.mean(), xgb_model_scores_test.std())
    print(msg)
    add_scores('RFE XGB', xgb_model_scores_train, xgb_model_scores_test)

    # ANN / MLP
    mlpc_model = MLPClassifier(random_state=SEED,max_iter=2000)
    params = {
            'activation':['relu','tanh'],
            'solver':['sgd','adam'],
            'learning_rate':['constant','adaptive'],
            'alpha':[0.0001, 0.001, 0.01]
            }
    mlpc_search = BayesSearchCV(mlpc_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    mlpc_search.fit(X_train_final, y_train_final)
    rfe_svm_best_mlpc_model = mlpc_search.best_estimator_
    mlpc_model_scores_train = cross_val_score(rfe_svm_best_mlpc_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - RFECV-SVM Tunned ANN Model Score: %f (%f)" % (mlpc_model_scores_train.mean(), mlpc_model_scores_train.std())
    print(msg)
    mlpc_model_scores_test = cross_val_score(rfe_svm_best_mlpc_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - RFECV-SVM Tunned ANN Model Score: %f (%f)" % (mlpc_model_scores_test.mean(), mlpc_model_scores_test.std())
    print(msg)
    add_scores('RFE ANN (MLP)', mlpc_model_scores_train, mlpc_model_scores_test)

    # Stacking
    base_learners = [
            ('LR', rfe_svm_best_lr_model),
            ('RF', rfe_svm_best_rf_model),
            ('ANN', rfe_svm_best_mlpc_model),
            ('XGB', rfe_svm_best_xgb_model),
            ]
    rfe_svm_stk_model = StackingClassifier(estimators=base_learners, final_estimator = lr_model, cv = cv_metric)
    rfe_svm_stk_model.fit(X_train_final, y_train_final)
    stk_model_scores_train = cross_val_score(rfe_svm_stk_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - RFECV-SVM Tunned Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_train.mean(), stk_model_scores_train.std())
    print(msg)
    stk_model_scores_test = cross_val_score(rfe_svm_stk_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - RFECV-SVM Tunned Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_test.mean(), stk_model_scores_test.std())
    print(msg)
    add_scores('RFE STACK', stk_model_scores_train, stk_model_scores_test)


    ################## Select Features using BorutaSHAP ##################
    print('Performing Feature Selection with BorutaSHAP')
    selector = BorutaShap(model = RandomForestClassifier(random_state=SEED), importance_measure = 'shap', classification = True)
    selector.fit(X=X_train_scaled, y=y_train, n_trials = 100, random_state=SEED)
    features_to_remove = selector.features_to_remove
    borutaSHAP_features = [column for column in X_train.columns if column not in features_to_remove]
    print('Best BorutaSHAP features: ', borutaSHAP_features)
    
    X_train_final = X_train_scaled.copy()
    X_test_final = X_test_scaled.copy()
    y_train_final = y_train.copy()
    y_test_final = y_test.copy()

    X_train_final = X_train_final[borutaSHAP_features]
    X_test_final = X_test_final[borutaSHAP_features]
    y_train_final = y_train_final
    y_test_final = y_test_final
    
    # Logistic regression
    lr_model = LogisticRegression(random_state=SEED)
    params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear','saga'],
            'class_weight':[None,'balanced']
        }
    lr_search = BayesSearchCV(lr_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    lr_search.fit(X_train_final, y_train_final)
    borutaSHAP_best_lr_model = lr_search.best_estimator_
    lr_model_scores_train = cross_val_score(borutaSHAP_best_lr_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - borutaSHAP Tunned LR Model Score: %f (%f)" % (lr_model_scores_train.mean(), lr_model_scores_train.std())
    print(msg)
    lr_model_scores_test = cross_val_score(borutaSHAP_best_lr_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - borutaSHAP Tunned LR Model Score: %f (%f)" % (lr_model_scores_test.mean(), lr_model_scores_test.std())
    print(msg)
    add_scores('BS LR', lr_model_scores_train, lr_model_scores_test)

    # Random forest
    rf_model = RandomForestClassifier(random_state=SEED)
    params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None,'sqrt','log2'],
            'criterion': ['gini', 'entropy'],
            'class_weight':[None,'balanced'],
            }
    rf_search = BayesSearchCV(rf_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    rf_search.fit(X_train_final, y_train_final)
    borutaSHAP_best_rf_model = rf_search.best_estimator_
    rf_model_scores_train = cross_val_score(borutaSHAP_best_rf_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - borutaSHAP Tunned RF Model Score: %f (%f)" % (rf_model_scores_train.mean(), rf_model_scores_train.std())
    print(msg)
    rf_model_scores_test = cross_val_score(borutaSHAP_best_rf_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - borutaSHAP Tunned RF Model Score: %f (%f)" % (rf_model_scores_test.mean(), rf_model_scores_test.std())
    print(msg)
    add_scores('BS RF', rf_model_scores_train, rf_model_scores_test)

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
    borutaSHAP_best_xgb_model = xgb_search.best_estimator_
    xgb_model_scores_train = cross_val_score(borutaSHAP_best_xgb_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - borutaSHAP Tunned XGB Model Score: %f (%f)" % (xgb_model_scores_train.mean(), xgb_model_scores_train.std())
    print(msg)
    xgb_model_scores_test = cross_val_score(borutaSHAP_best_xgb_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - borutaSHAP Tunned XGB Model Score: %f (%f)" % (xgb_model_scores_test.mean(), xgb_model_scores_test.std())
    print(msg)
    add_scores('BS XGB', xgb_model_scores_train, xgb_model_scores_test)

    # ANN / MLP
    mlpc_model = MLPClassifier(random_state=SEED,max_iter=2000)
    params = {
            #'hidden_layer_sizes': [(50,),(100,),(50,50),(100,50,25)],
            'activation':['relu','tanh'],
            'solver':['sgd','adam'],
            'learning_rate':['constant','adaptive'],
            'alpha':[0.0001, 0.001, 0.01]
            }
    mlpc_search = BayesSearchCV(mlpc_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    mlpc_search.fit(X_train_final, y_train_final)
    borutaSHAP_best_mlpc_model = mlpc_search.best_estimator_
    mlpc_model_scores_train = cross_val_score(borutaSHAP_best_mlpc_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - borutaSHAP Tunned ANN Model Score: %f (%f)" % (mlpc_model_scores_train.mean(), mlpc_model_scores_train.std())
    print(msg)
    mlpc_model_scores_test = cross_val_score(borutaSHAP_best_mlpc_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - borutaSHAP Tunned ANN Model Score: %f (%f)" % (mlpc_model_scores_test.mean(), mlpc_model_scores_test.std())
    print(msg)
    add_scores('BS ANN (MLP)', mlpc_model_scores_train, mlpc_model_scores_test)

    # Stacking
    base_learners = [
            ('LR', borutaSHAP_best_lr_model),
            ('RF', borutaSHAP_best_rf_model),
            ('ANN', borutaSHAP_best_mlpc_model),
            ('XGB', borutaSHAP_best_xgb_model),
            ]
    borutaSHAP_stk_model = StackingClassifier(estimators=base_learners, final_estimator = lr_model, cv = cv_metric)
    borutaSHAP_stk_model.fit(X_train_final, y_train_final)
    stk_model_scores_train = cross_val_score(borutaSHAP_stk_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - BorutaSHAP Tunned Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_train.mean(), stk_model_scores_train.std())
    print(msg)
    stk_model_scores_test = cross_val_score(borutaSHAP_stk_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - BorutaSHAP Tunned Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_test.mean(), stk_model_scores_test.std())
    print(msg)
    add_scores('BS STACK', stk_model_scores_train, stk_model_scores_test)


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

    # Logistic regression
    lr_model = LogisticRegression(random_state=SEED)
    params = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear','saga'],
            'class_weight':[None,'balanced']
        }
    lr_search = BayesSearchCV(lr_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    lr_search.fit(X_train_final, y_train_final)
    lasso_best_lr_model = lr_search.best_estimator_
    lr_model_scores_train = cross_val_score(lasso_best_lr_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - LASSO Tunned LR Model Score: %f (%f)" % (lr_model_scores_train.mean(), lr_model_scores_train.std())
    print(msg)
    lr_model_scores_test = cross_val_score(lasso_best_lr_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - LASSO Tunned LR Model Score: %f (%f)" % (lr_model_scores_test.mean(), lr_model_scores_test.std())
    print(msg)
    add_scores('LASSO LR', lr_model_scores_train, lr_model_scores_test)

    # Random forest
    rf_model = RandomForestClassifier(random_state=SEED)
    params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None,'sqrt','log2'],
            'criterion': ['gini', 'entropy'],
            'class_weight':[None,'balanced'],
            }
    rf_search = BayesSearchCV(rf_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    rf_search.fit(X_train_final, y_train_final)
    lasso_best_rf_model = rf_search.best_estimator_
    rf_model_scores_train = cross_val_score(lasso_best_rf_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - LASSO Tunned RF Model Score: %f (%f)" % (rf_model_scores_train.mean(), rf_model_scores_train.std())
    print(msg)
    rf_model_scores_test = cross_val_score(lasso_best_rf_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - LASSO Tunned RF Model Score: %f (%f)" % (rf_model_scores_test.mean(), rf_model_scores_test.std())
    print(msg)
    add_scores('LASSO RF', rf_model_scores_train, rf_model_scores_test)

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

    # ANN / MLP
    mlpc_model = MLPClassifier(random_state=SEED,max_iter=2000)
    params = {
            #'hidden_layer_sizes': [(50,),(100,),(50,50),(100,50,25)],
            'activation':['relu','tanh'],
            'solver':['sgd','adam'],
            'learning_rate':['constant','adaptive'],
            'alpha':[0.0001, 0.001, 0.01]
            }
    mlpc_search = BayesSearchCV(mlpc_model, params, scoring = scoring_metric, cv = cv_metric, random_state = SEED)
    mlpc_search.fit(X_train_final, y_train_final)
    lasso_best_mlpc_model = mlpc_search.best_estimator_
    mlpc_model_scores_train = cross_val_score(lasso_best_mlpc_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - LASSO Tunned ANN Model Score: %f (%f)" % (mlpc_model_scores_train.mean(), mlpc_model_scores_train.std())
    print(msg)
    mlpc_model_scores_test = cross_val_score(lasso_best_mlpc_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - LASSO Tunned ANN Model Score: %f (%f)" % (mlpc_model_scores_test.mean(), mlpc_model_scores_test.std())
    print(msg)
    add_scores('LASSO ANN (MLPClassifier)', mlpc_model_scores_train, mlpc_model_scores_test)

    # Stacking
    base_learners = [
            ('LR', lasso_best_lr_model),
            ('RF', lasso_best_rf_model),
            ('ANN', lasso_best_mlpc_model),
            ('XGB', lasso_best_xgb_model),
            ]
    LASSO_stk_model = StackingClassifier(estimators=base_learners, final_estimator = lr_model, cv = cv_metric)
    LASSO_stk_model.fit(X_train_final, y_train_final)
    stk_model_scores_train = cross_val_score(LASSO_stk_model, X_train_final, y_train_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TRAINING SET - LASSO Tunned Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_train.mean(), stk_model_scores_train.std())
    print(msg)
    stk_model_scores_test = cross_val_score(LASSO_stk_model, X_test_final, y_test_final, cv = cv_metric, scoring=scoring_metric)
    msg = "TEST SET - LASSO Tunned Stacking Classifier Model Score: %f (%f)" % (stk_model_scores_test.mean(), stk_model_scores_test.std())
    print(msg)
    add_scores('LASSO STACK', stk_model_scores_train, stk_model_scores_test)


    ##### Saving Model Scores #####
    df_scores = pd.DataFrame(scores)
    # Save DataFrame to Excel
    file_name = f'model_scores_{sheet_name}_{SEED}_{split_ratio}_ROCAUC.xlsx'  # ROCAUC is the scoring metric 
    df_scores.to_excel(file_name, index=False)
    print(f"Scores successfully saved to {file_name}")
    

    ##### Saving Model Features Selected #####
    # Create a dictionary with column names as keys and lists as values for saving features
    data_feats = {
        'Manual': manually_selected_features,
        'RFE-SVM': rfe_svm_features,
        'BorutaSHHAP': borutaSHAP_features,
        'LASSO': lasso_features
    }
    df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data_feats.items()]))
    # Save DataFrame to Excel
    file_name_feats = f'features_{sheet_name}_{SEED}_{split_ratio}_ROCAUC.xlsx'
    df.to_excel(file_name_feats, index=False)
    print(f"Table written to {file_name_feats}")