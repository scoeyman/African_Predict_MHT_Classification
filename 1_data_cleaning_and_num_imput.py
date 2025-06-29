# Import Libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
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

# Set Defined Parameters
SEED = 42

# Load Data
file_path = r"C:\Users\scoeyman\Desktop\af predict manual cat imputing\last time (papers)"
file_name = 'AP_Baseline_Richardson.xlsx'

# Combine file_path and file_name
full_path = f"{file_path}\\{file_name}"

# Load the Excel file
df = pd.read_excel(full_path)
df_cleaned = df.copy() 

# Create Single Variable "SBP" from different SBP measures
sbp = pd.DataFrame((df_cleaned['l_sbp_1']+df_cleaned['l_sbp_2']
                            +df_cleaned['r_sbp_1']+df_cleaned['r_sbp_2']
                            +df_cleaned['sphygmocor_sbp1']+df_cleaned['sphygmocor_sbp2']
                            +df_cleaned['csbp_1']+df_cleaned['csbp_2']
                            +df_cleaned['l_osbp']+df_cleaned['r_osbp'])/10, columns=['sbp'])
df_cleaned = pd.concat([df_cleaned,sbp],axis = 1)
df_cleaned = df_cleaned.drop(['l_sbp_1','l_sbp_2','r_sbp_1','r_sbp_2','sphygmocor_sbp1',
                  'sphygmocor_sbp2','csbp_1','csbp_2','l_osbp','r_osbp'],axis=1)

# Create Single Variable "DBP" from different DBP measures
dbp = pd.DataFrame((df_cleaned['l_dbp_1']+df_cleaned['l_dbp_2']
                            +df_cleaned['r_dbp_1']+df_cleaned['r_dbp_2']
                            +df_cleaned['sphygmocor_dbp1']+df_cleaned['sphygmocor_dbp2']
                            +df_cleaned['cdbp_1']+df_cleaned['cdbp_2']
                            +df_cleaned['l_odbp']+df_cleaned['r_odbp'])/10, columns=['dbp'])
df_cleaned = pd.concat([df_cleaned,dbp],axis = 1)
df_cleaned = df_cleaned.drop(['l_dbp_1','l_dbp_2','r_dbp_1','r_dbp_2','sphygmocor_dbp1',
                  'sphygmocor_dbp2','cdbp_1','cdbp_2','l_odbp','r_odbp'],axis=1)

# Create average hr metric
hr = pd.DataFrame((df_cleaned['l_hr_1']+df_cleaned['l_hr_2']
                    +df_cleaned['r_hr_1']+df_cleaned['r_hr_2']
                    +df_cleaned['sphyg_hr1']+df_cleaned['sphyg_hr2'])/6, columns=['hr'])
df_cleaned = pd.concat([df_cleaned,hr],axis = 1)
df_cleaned = df_cleaned.drop(['l_hr_1','l_hr_2','r_hr_1','r_hr_2','sphyg_hr1','sphyg_hr2'],axis=1)

# Create average cpp metric
cpp = pd.DataFrame((df_cleaned['cpp1']+df_cleaned['cpp2'])/2, columns=['cpp'])
df_cleaned = pd.concat([df_cleaned,cpp],axis = 1)
df_cleaned = df_cleaned.drop(['cpp1','cpp2'],axis=1)

# Create average map metric
map = df_cleaned['clinic_map'].rename('map')
df_cleaned = pd.concat([df_cleaned,map],axis = 1)
df_cleaned = df_cleaned.drop(['clinic_map'],axis=1)

# Create average map_syst metric
cmp = pd.DataFrame((df_cleaned['map_syst_1']+df_cleaned['map_syst_2'])/2, columns=['map_syst'])
df_cleaned = pd.concat([df_cleaned,cmp],axis = 1)
df_cleaned = df_cleaned.drop(['map_syst_1','map_syst_2'],axis=1)

# Create average map_dia metric
cmp = pd.DataFrame((df_cleaned['map_dia_1']+df_cleaned['map_dia_2'])/2, columns=['map_dia'])
df_cleaned = pd.concat([df_cleaned,cmp],axis = 1)
df_cleaned = df_cleaned.drop(['map_dia_1','map_dia_2'],axis=1)

# Create average cmp metric
cmp = pd.DataFrame((df_cleaned['cmp_1']+df_cleaned['cmp_2'])/2, columns=['cmp'])
df_cleaned = pd.concat([df_cleaned,cmp],axis = 1)
df_cleaned = df_cleaned.drop(['cmp_1','cmp_2'],axis=1)

# Create average pp metric
pp = pd.DataFrame((df_cleaned['pulse_pressure']+df_cleaned['l_pp']+df_cleaned['r_pp']/3),columns=['pp'])
df_cleaned = pd.concat([df_cleaned,pp],axis = 1)
df_cleaned = df_cleaned.drop(['pulse_pressure','l_pp','r_pp'],axis=1)

# Create average ap metric
ap = pd.DataFrame((df_cleaned['ap_1']+df_cleaned['ap_2'])/2, columns=['ap'])
df_cleaned = pd.concat([df_cleaned,ap],axis = 1)
df_cleaned = df_cleaned.drop(['ap_1','ap_2'],axis=1)

# Create average cai metric
cai = pd.DataFrame((df_cleaned['cai_1']+df_cleaned['cai_2'])/2, columns=['cai'])
df_cleaned = pd.concat([df_cleaned,cai],axis = 1)
df_cleaned = df_cleaned.drop(['cai_1','cai_2'],axis=1)

# Create average imtn_mean metric
imtn_mean = pd.DataFrame((df_cleaned['l_imtn_mean']+df_cleaned['r_imtn_mean'])/2, columns=['imtn_mean'])
df_cleaned = pd.concat([df_cleaned,imtn_mean],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtn_mean','r_imtn_mean'],axis=1)

# Create average imtn_min metric
imtn_min = pd.DataFrame((df_cleaned['l_imtn_min']+df_cleaned['r_imtn_min'])/2, columns=['imtn_min'])
df_cleaned = pd.concat([df_cleaned,imtn_min],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtn_min','r_imtn_min'],axis=1)

# Create average imtn_max metric
imtn_max = pd.DataFrame((df_cleaned['l_imtn_max']+df_cleaned['r_imtn_max'])/2, columns=['imtn_max'])
df_cleaned = pd.concat([df_cleaned,imtn_max],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtn_max','r_imtn_max'],axis=1)

# Create average imtn_std metric
imtn_std = pd.DataFrame((df_cleaned['l_imtn_std']+df_cleaned['r_imtn_std'])/2, columns=['imtn_std'])
df_cleaned = pd.concat([df_cleaned,imtn_std],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtn_std','r_imtn_std'],axis=1)

# Create average imtn_length metric
imtn_length = pd.DataFrame((df_cleaned['l_imtn_length']+df_cleaned['r_imtn_length'])/2, columns=['imtn_length'])
df_cleaned = pd.concat([df_cleaned,imtn_length],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtn_length','r_imtn_length'],axis=1)

# Create average imtf_mean metric
imtf_mean = pd.DataFrame((df_cleaned['l_imtf_mean']+df_cleaned['r_imtf_mean'])/2, columns=['imtf_mean'])
df_cleaned = pd.concat([df_cleaned,imtf_mean],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtf_mean','r_imtf_mean'],axis=1)

# Create average imtf_min metric
imtf_min = pd.DataFrame((df_cleaned['l_imtf_min']+df_cleaned['r_imtf_min'])/2, columns=['imtf_min'])
df_cleaned = pd.concat([df_cleaned,imtf_min],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtf_min','r_imtf_min'],axis=1)

# Create average imtf_max metric
imtf_max = pd.DataFrame((df_cleaned['l_imtf_max']+df_cleaned['r_imtf_max'])/2, columns=['imtf_max'])
df_cleaned = pd.concat([df_cleaned,imtf_max],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtf_max','r_imtf_max'],axis=1)

# Create average imtf_std metric
imtf_std = pd.DataFrame((df_cleaned['l_imtf_std']+df_cleaned['r_imtf_std'])/2, columns=['imtf_std'])
df_cleaned = pd.concat([df_cleaned,imtf_std],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtf_std','r_imtf_std'],axis=1)

# Create average imtf_length metric
imtf_length = pd.DataFrame((df_cleaned['l_imtf_length']+df_cleaned['r_imtf_length'])/2, columns=['imtf_length'])
df_cleaned = pd.concat([df_cleaned,imtf_length],axis = 1)
df_cleaned = df_cleaned.drop(['l_imtf_length','r_imtf_length'],axis=1)

# Create average ld_mean metric
ld_mean = pd.DataFrame((df_cleaned['l_ld_mean']+df_cleaned['r_ld_mean'])/2, columns=['ld_mean'])
df_cleaned = pd.concat([df_cleaned,ld_mean],axis = 1)
df_cleaned = df_cleaned.drop(['l_ld_mean','r_ld_mean'],axis=1)

# Create average ld_min metric
ld_min = pd.DataFrame((df_cleaned['l_ld_min']+df_cleaned['r_ld_min'])/2, columns=['ld_min'])
df_cleaned = pd.concat([df_cleaned,ld_min],axis = 1)
df_cleaned = df_cleaned.drop(['l_ld_min','r_ld_min'],axis=1)

# Create average ld_max metric
ld_max = pd.DataFrame((df_cleaned['l_ld_max']+df_cleaned['r_ld_max'])/2, columns=['ld_max'])
df_cleaned = pd.concat([df_cleaned,ld_max],axis = 1)
df_cleaned = df_cleaned.drop(['l_ld_max','r_ld_max'],axis=1)

# Create average ld_std metric
ld_std = pd.DataFrame((df_cleaned['l_ld_std']+df_cleaned['r_ld_std'])/2, columns=['ld_std'])
df_cleaned = pd.concat([df_cleaned,ld_std],axis = 1)
df_cleaned = df_cleaned.drop(['l_ld_std','r_ld_std'],axis=1)

# Create average ld_legth metric
ld_legth = pd.DataFrame((df_cleaned['l_ld_legth']+df_cleaned['r_ld_legth'])/2, columns=['ld_legth'])
df_cleaned = pd.concat([df_cleaned,ld_legth],axis = 1)
df_cleaned = df_cleaned.drop(['l_ld_legth','r_ld_legth'],axis=1)

# Create average ad_mean metric
ad_mean = pd.DataFrame((df_cleaned['l_ad_mean']+df_cleaned['r_ad_mean'])/2, columns=['ad_mean'])
df_cleaned = pd.concat([df_cleaned,ad_mean],axis = 1)
df_cleaned = df_cleaned.drop(['l_ad_mean','r_ad_mean'],axis=1)

# Create average ad_min metric
ad_min = pd.DataFrame((df_cleaned['l_ad_min']+df_cleaned['r_ad_min'])/2, columns=['ad_min'])
df_cleaned = pd.concat([df_cleaned,ad_min],axis = 1)
df_cleaned = df_cleaned.drop(['l_ad_min','r_ad_min'],axis=1)

# Create average ad_max metric
ad_max = pd.DataFrame((df_cleaned['l_ad_max']+df_cleaned['r_ad_max'])/2, columns=['ad_max'])
df_cleaned = pd.concat([df_cleaned,ad_max],axis = 1)
df_cleaned = df_cleaned.drop(['l_ad_max','r_ad_max'],axis=1)

# Create average ad_std metric
ad_std = pd.DataFrame((df_cleaned['l_ad_std']+df_cleaned['r_ad_std'])/2, columns=['ad_std'])
df_cleaned = pd.concat([df_cleaned,ad_std],axis = 1)
df_cleaned = df_cleaned.drop(['l_ad_std','r_ad_std'],axis=1)

# Create average ad_legth metric
ad_length = pd.DataFrame((df_cleaned['l_ad_length']+df_cleaned['r_ad_length'])/2, columns=['ad_length'])
df_cleaned = pd.concat([df_cleaned,ad_length],axis = 1)
df_cleaned = df_cleaned.drop(['l_ad_length','r_ad_length'],axis=1)

# Create average rough_near metric
rough_near = pd.DataFrame((df_cleaned['l_rough_near']+df_cleaned['r_rough_near'])/2, columns=['rough_near'])
df_cleaned = pd.concat([df_cleaned,rough_near],axis = 1)
df_cleaned = df_cleaned.drop(['l_rough_near','r_rough_near'],axis=1)

# Create average rough_far metric
rough_far = pd.DataFrame((df_cleaned['l_rough_far']+df_cleaned['r_rough_far'])/2, columns=['rough_far'])
df_cleaned = pd.concat([df_cleaned,rough_far],axis = 1)
df_cleaned = df_cleaned.drop(['l_rough_far','r_rough_far'],axis=1)

# Create average cca_ps metric
cca_ps = pd.DataFrame((df_cleaned['l_cca_ps']+df_cleaned['r_cca_ps'])/2, columns=['cca_ps'])
df_cleaned = pd.concat([df_cleaned,cca_ps],axis = 1)
df_cleaned = df_cleaned.drop(['l_cca_ps','r_cca_ps'],axis=1)

# Create average cca_ed metric
cca_ed = pd.DataFrame((df_cleaned['l_cca_ed']+df_cleaned['r_cca_ed'])/2, columns=['cca_ed'])
df_cleaned = pd.concat([df_cleaned,cca_ed],axis = 1)
df_cleaned = df_cleaned.drop(['l_cca_ed','r_cca_ed'],axis=1)

# Create average ica_ps metric
ica_ps = pd.DataFrame((df_cleaned['l_ica_ps']+df_cleaned['r_ica_ps'])/2, columns=['ica_ps'])
df_cleaned = pd.concat([df_cleaned,ica_ps],axis = 1)
df_cleaned = df_cleaned.drop(['l_ica_ps','r_ica_ps'],axis=1)

# Create average ica_ed metric
ica_ed = pd.DataFrame((df_cleaned['l_ica_ed']+df_cleaned['r_ica_ed'])/2, columns=['ica_ed'])
df_cleaned = pd.concat([df_cleaned,ica_ed],axis = 1)
df_cleaned = df_cleaned.drop(['l_ica_ed','r_ica_ed'],axis=1)

# Create average cswa metric
cswa = pd.DataFrame((df_cleaned['lcswa']+df_cleaned['rcswa'])/2, columns=['cswa'])
df_cleaned = pd.concat([df_cleaned,cswa],axis = 1)
df_cleaned = df_cleaned.drop(['lcswa','rcswa'],axis=1)

# Create average pwv metric
pwv = pd.DataFrame((df_cleaned['pwv1']+df_cleaned['pwv2'])/2, columns=['pwv'])
df_cleaned = pd.concat([df_cleaned,pwv],axis = 1)
df_cleaned = df_cleaned.drop(['pwv1','pwv2'],axis=1)

# Create average acimt metric
acimt = pd.DataFrame((df_cleaned['l_acimt']+df_cleaned['r_acimt'])/2, columns=['acimt'])
df_cleaned = pd.concat([df_cleaned,acimt],axis = 1)
df_cleaned = df_cleaned.drop(['l_acimt','r_acimt'],axis=1)

# Create average delta_ld metric
delta_ld = pd.DataFrame((df_cleaned['l_delta_ld']+df_cleaned['r_delta_ld'])/2, columns=['delta_ld'])
df_cleaned = pd.concat([df_cleaned,delta_ld],axis = 1)
df_cleaned = df_cleaned.drop(['l_delta_ld','r_delta_ld'],axis=1)

# Create average strain metric
strain = pd.DataFrame((df_cleaned['l_strain']+df_cleaned['r_strain'])/2, columns=['strain'])
df_cleaned = pd.concat([df_cleaned,strain],axis = 1)
df_cleaned = df_cleaned.drop(['l_strain','r_strain'],axis=1)

# Create average distensibility metric
distensibility = pd.DataFrame((df_cleaned['l_distensibility']+df_cleaned['r_distensibility'])/2, columns=['distensibility'])
df_cleaned = pd.concat([df_cleaned,distensibility],axis = 1)
df_cleaned = df_cleaned.drop(['l_distensibility','r_distensibility'],axis=1)

# Create average compliance metric
compliance = pd.DataFrame((df_cleaned['l_compliance']+df_cleaned['r_compliance'])/2, columns=['compliance'])
df_cleaned = pd.concat([df_cleaned,compliance],axis = 1)
df_cleaned = df_cleaned.drop(['l_compliance','r_compliance'],axis=1)

# Create average bsi metric
bsi = pd.DataFrame((df_cleaned['l_bsi']+df_cleaned['r_bsi'])/2, columns=['bsi'])
df_cleaned = pd.concat([df_cleaned,bsi],axis = 1)
df_cleaned = df_cleaned.drop(['l_bsi','r_bsi'],axis=1)

# Create average pem metric
pem = pd.DataFrame((df_cleaned['l_pem']+df_cleaned['r_pem'])/2, columns=['pem'])
df_cleaned = pd.concat([df_cleaned,pem],axis = 1)
df_cleaned = df_cleaned.drop(['l_pem','r_pem'],axis=1)

# Create average yem metric
yem = pd.DataFrame((df_cleaned['l_yem']+df_cleaned['r_yem'])/2, columns=['yem'])
df_cleaned = pd.concat([df_cleaned,yem],axis = 1)
df_cleaned = df_cleaned.drop(['l_yem','r_yem'],axis=1)

# Create prehypertensive metric
df_cleaned['prehypertensive'] = np.nan
df_cleaned.loc[(df_cleaned['sbp'] <= 119) & (df_cleaned['dbp'] <= 79), 'prehypertensive'] = 0
df_cleaned.loc[(df_cleaned['sbp'] < 140) & (df_cleaned['sbp'] > 119) | (df_cleaned['dbp'] < 90) & (df_cleaned['dbp'] > 79), 'prehypertensive'] = 1

# Create obese metric
df_cleaned['obese'] = np.nan
df_cleaned.loc[(df_cleaned['bmi'] <= 25), 'obese'] = 0
df_cleaned.loc[(df_cleaned['bmi'] > 25), 'obese'] = 1

# Create left ventrical hypertrophy metric
df_cleaned['lvh'] = np.nan
df_cleaned.loc[((df_cleaned['ilvm_cube'] <= 115) & (df_cleaned['sex'] == 1)) | ((df_cleaned['ilvm_cube'] <= 95) & (df_cleaned['sex'] == 0)), 'lvh'] = 0
df_cleaned.loc[((df_cleaned['ilvm_cube'] > 115) & (df_cleaned['sex'] == 1)) | ((df_cleaned['ilvm_cube'] > 95) & (df_cleaned['sex'] == 0)), 'lvh'] = 1

# Create sedentary metric
df_cleaned['sedentary'] = np.nan
df_cleaned.loc[df_cleaned['vig_work_met_min'] + df_cleaned['moderate_work_met_minutes'] + df_cleaned['travel_met_minutes'] + df_cleaned['vig_exercise_met'] + df_cleaned['moderate_exercise_met'] >= 600,'sedentary'] = 0
df_cleaned.loc[df_cleaned['vig_work_met_min'] + df_cleaned['moderate_work_met_minutes'] + df_cleaned['travel_met_minutes'] + df_cleaned['vig_exercise_met'] + df_cleaned['moderate_exercise_met'] < 600,'sedentary'] = 1

# Create smoker metric
df_cleaned['smoker'] = np.nan
df_cleaned.loc[(df_cleaned['cotinine'] < 11) | (df_cleaned['smoke'] == 0),'smoker'] = 0
df_cleaned.loc[(df_cleaned['cotinine'] >= 11) & (df_cleaned['smoke'] == 1),'smoker'] = 1

# Create excessive alcohol metric
df_cleaned['excessive_alcohol'] = np.nan
df_cleaned.loc[(df_cleaned['ggt'] < 49) | (df_cleaned['alcohol'] == 0),'excessive_alcohol'] = 0
df_cleaned.loc[(df_cleaned['ggt'] >= 49) & (df_cleaned['alcohol'] == 1),'excessive_alcohol'] = 1

# Create dyslipidemic metric
df_cleaned['dyslipidemic'] = np.nan
df_cleaned.loc[df_cleaned['ldl'] < 3.5,'dyslipidemic'] = 0
df_cleaned.loc[df_cleaned['ldl'] >= 3.5,'dyslipidemic'] = 1

# Define the columns to remove by their names (AMBULATORY ONES + AHA ones + old glucose)
columns_to_remove = ['apparatus', 'infl_24', 'nr_infl', 'day_infl',
                     'perc_succ_infl', 'seventy_perc_yn', 'night_infl',
                     'day_inflation_complete', 'night_inflation_complete',
                     'abpm_24_s', 'abpm_24_d', 'abpm_24_pp', 'map_24h',
                     'abpm_24_ht', 'abpm_d_sbp', 'abpm_d_dbp', 'abpm_d_pp',
                     'map_day', 'abpm_d_ht', 'abpm_n_sbp', 'abpm_n_dbp',
                     'abpm_n_pp', 'map_night', 'abpm_n_ht', 'percent_dip',
                     'dipper', 'masked_ht', 'pulse_24', 'pulse_d', 'pulse_n',
                     'aha_ht', 'aha_sht', 'aha_mht', 'aha_wcht','aha_abpm_ht',
                     'smoke', 'alcohol', 'pulse_night', 'percent_dip', 'travel_met_minutes', 'glu']

# Remove the specified columns if they exist
df_cleaned.drop(columns=[col for col in columns_to_remove if col in df_cleaned.columns], inplace=True)

# Drop Rows Missing Label Values
df_cleaned.dropna(subset = ['abpm_overall_ht'], inplace = True)

# Drop Rows with Hypertensive Patients
df_cleaned = df_cleaned[df_cleaned['clinic_bp_status'] != 1]

# Remove Columns Missing More Than 10% of Data
missing_percent_columns = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100
columns_to_drop = missing_percent_columns[missing_percent_columns > 10].index.tolist()
df_cleaned = df_cleaned.drop(columns=columns_to_drop)

# Remove Rows Missing More Than 10% of Data
missing_percent_rows = (df_cleaned.isnull().sum(axis=1) / len(df_cleaned.columns)) * 100
rows_to_drop = missing_percent_rows[missing_percent_rows > 10].index.tolist()
df_cleaned = df_cleaned.drop(index=rows_to_drop)

# Remove Columns With Only 1 Value
unique_counts = df_cleaned.nunique()
unique_columns = unique_counts[unique_counts == 1].index.tolist()

# Defining categoric features (only numerical ones will be imputed in this code)
categoric_features = ['sex','ethnicity','s_meds_a','s_meds_b','s_meds_d','s_meds_g','s_meds_h','s_meds_j','s_meds_l','s_meds_m',
                      's_meds_n','s_meds_p','s_meds_r','s_meds_s','s_meds_v','s_meds_c','s_meds_c_cardiac','s_meds_c_ht','s_meds_c_ht',
                      's_meds_c_diur','s_meds_c_bb','s_meds_c_bb','s_meds_c_ccb','s_meds_c_ren','s_meds_c_lipid','s_meds_c_other',
                      'prehypertensive','obese','lvh','sedentary','smoker','excessive_alcohol','dyslipidemic','clinic_bp_status', 
                      'ses_skill', 'ses_education', 'ses_income', 'ses_score', 'ses_class', 'bp_grade']
                      
# Ensure no duplicate columns in categoric_features
categoric_features = list(dict.fromkeys(categoric_features))

# Selecting categorical data using the list of categorical features
categorical_data = df_cleaned[categoric_features]

# Selecting numeric data by excluding the categorical columns
numeric_data = df_cleaned.drop(columns=categoric_features).select_dtypes(include=[np.number])

# Extracting the label series (assuming 'abpm_overall_ht' is the label)
label_series = df_cleaned['abpm_overall_ht']

# Convert identified columns to numeric (coerce errors to NaN for non-convertible values)
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# Display the first few rows of each to verify
print("Numeric Data Shape:", numeric_data.shape)
print("Categorical Data Shape:", categorical_data.shape)

# Print the identified numeric features
print("Numeric Features:")
print(numeric_data.columns.tolist())

# Numeric Imputing
seed = SEED
iter_imp_numeric = IterativeImputer(estimator=GradientBoostingRegressor(random_state=seed), max_iter=10, random_state=seed)
imputed_data = iter_imp_numeric.fit_transform(numeric_data)

# Check the shape of imputed data
print("Shape of imputed_data:", imputed_data.shape)
print("Expected shape (based on numeric_data):", numeric_data.shape)

# Ensure the shapes match before creating DataFrame
if imputed_data.shape[1] == numeric_data.shape[1]:
    imputed_data = pd.DataFrame(imputed_data, columns=numeric_data.columns, index=numeric_data.index)
else:
    # Create a DataFrame from the imputed data
    imputed_data = pd.DataFrame(imputed_data, columns=numeric_data.columns[:imputed_data.shape[1]], index=numeric_data.index)

    # Identify missing columns
    missing_columns = set(numeric_data.columns) - set(imputed_data.columns)
    print("Missing columns:", missing_columns)

    # Manually add back missing columns if any
    for col in missing_columns:
        imputed_data[col] = numeric_data[col]

    # Ensure the columns are in the same order as the original numeric_data
    imputed_data = imputed_data[numeric_data.columns]

    # Raise an error if shapes still do not match
    if imputed_data.shape[1] != numeric_data.shape[1]:
        raise ValueError("Shape mismatch between imputed data and original numeric data")

# Check for remaining missing values in the imputed data and fill them with a suitable strategy
if imputed_data.isna().sum().sum() != 0:
    # Fill remaining NaNs with the mean of each column
    imputed_data = imputed_data.fillna(imputed_data.mean())

# Print the imputed numeric data
print("\nImputed Numeric Data Shape:")
print(imputed_data.shape)

# Combine imputed numeric data with categorical data and label
final_df = pd.concat([imputed_data, categorical_data, label_series], axis=1)

# Remove Columns Missing More Than 10% of Data
missing_percent_columns = (final_df.isnull().sum() / len(final_df)) * 100
columns_to_drop = missing_percent_columns[missing_percent_columns > 10].index.tolist()
final_df = final_df.drop(columns=columns_to_drop)

# Remove Rows Missing More Than 10% of Data
missing_percent_rows = (final_df.isnull().sum(axis=1) / len(final_df.columns)) * 100
rows_to_drop = missing_percent_rows[missing_percent_rows > 10].index.tolist()
final_df = final_df.drop(index=rows_to_drop)

# Remove Columns With Only 1 Value
unique_counts = final_df.nunique()
unique_columns = unique_counts[unique_counts == 1].index.tolist()
final_df = final_df.drop(columns=unique_columns)

# Save to Excel file
output_file_path = 'cleaned_data_with_imputation.xlsx'
final_df.to_excel(output_file_path, index=False)
print(f"Cleaned data saved to {output_file_path}")

# Imputation Testing
# Check for remaining missing values in the imputed data
assert imputed_data.isna().sum().sum() == 0, "There are still missing values in the imputed data."
print("Imputation completed successfully.")
