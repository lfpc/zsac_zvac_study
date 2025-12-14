import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import warnings
# Suppress all warnings (including those from matplotlib, sklearn, etc.)
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Select the dataset/feature mode: 'background', 'agnostic', 'sequence'
# 'background': Uses a comprehensive set of engineered features.
# 'agnostic': Uses a subset of features and allows for further selection via filtering flags.
# 'sequence': Combines base features with specific exposure sequence data.

# Feature Engineering/Selection Flags
SCALE = True

# Flags for 'agnostic' mode - Will be tested individually
K_MI = 20  # Number of features to select with Mutual Information
K_RF = 20  # Number of features to select with RF Importance
CORR_THRESHOLD = 0.9
PCA_COMPONENTS = 0.95

# Test configurations for different feature selection methods
TEST_CONFIGS = {
    'no_filtering': {
        'APPLY_PCA': False,
        'FILTER_MI': False,
        'FILTER_CORRELATION': False,
        'FILTER_RF': False
    },
    'pca_only': {
        'APPLY_PCA': True,
        'FILTER_MI': False,
        'FILTER_CORRELATION': False,
        'FILTER_RF': False
    },
    'mi_only': {
        'APPLY_PCA': False,
        'FILTER_MI': True,
        'FILTER_CORRELATION': False,
        'FILTER_RF': False
    },
    'correlation_only': {
        'APPLY_PCA': False,
        'FILTER_MI': False,
        'FILTER_CORRELATION': True,
        'FILTER_RF': False
    },
    'rf_only': {
        'APPLY_PCA': False,
        'FILTER_MI': False,
        'FILTER_CORRELATION': False,
        'FILTER_RF': True
    }
}

# Model Training Config
N_SPLITS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Data files
DATA_DIR = 'data'
DATA_FILE = join(DATA_DIR, 'ZSAC_ZVAC_Collaboration_data_v3_20241121.csv')
DATA_FILE_SEQUENCE = join(DATA_DIR, 'ZSAC_ZVAC_Collaboration_sequence_data_20250124.csv')

# Constants
REF_DATE = pd.to_datetime('2022-01-01')
TARGET_COL = 'final_outcome_amp'


# --- HELPER & UTILITY FUNCTIONS ---

def filter_data_date(data, reference_date=REF_DATE):
    """For each person (record_id), get only the row with the visit date closest to reference_date."""
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    data = data.dropna(subset=['visit_date'], ignore_index=True)
    data = data[data['visit_date'] < reference_date]
    return data.sort_values(by=['record_id', 'visit_date'], ascending=[True, False]).drop_duplicates(subset=['record_id'], keep='first')

def scale(data):
    """Scales data if the global SCALE flag is True."""
    if SCALE:
        data = (data - data.mean()) / data.std()
    return data

def filter_by_mutual_information(X, y, k='all'):
    """Filters features based on mutual information with the target variable."""
    print(f"Applying Mutual Information filter (k={k})...")
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return X[selected_features]

def filter_by_correlation(X, threshold=0.9):
    """Filters features based on pairwise correlation."""
    print(f"Applying Correlation filter (threshold={threshold})...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping {len(to_drop)} features due to high correlation: {to_drop}")
    return X.drop(to_drop, axis=1)

def apply_pca(X, n_components=None):
    """Applies Principal Component Analysis (PCA) to the feature matrix."""
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA applied. Explained variance ratio with {pca.n_components_} components: {pca.explained_variance_ratio_.sum():.4f}")
    pca_cols = [f'PC_{i+1}' for i in range(pca.n_components_)]
    return pd.DataFrame(X_pca, index=X.index, columns=pca_cols)

def filter_by_rf_importance(X, y, k=10, **rf_params):
    """Filters features based on Random Forest feature importance."""
    print(f"Applying Random Forest Importance filter (top {k} features)...")
    default_rf_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    default_rf_params.update(rf_params)
    rf = RandomForestClassifier(**default_rf_params)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_k_features = importances.nlargest(k).index
    return X[top_k_features]


# --- FEATURE CREATION FUNCTIONS ---

def create_features_background(df):
    """Creates the comprehensive feature set from the 'background' analysis."""
    data = pd.DataFrame(index=df.index)
    data['study'] = df['study'].map({df['study'].unique()[0]: 0, df['study'].unique()[1]: 1})
    data['final_outcome_amp'] = df['final_outcome_amp']
    #data['index_date'] = (REF_DATE - pd.to_datetime(df['index_date'], format='%Y-%m-%d')).dt.days

    most_recent_vaccine = df[['vaccine_date_4', 'vaccine_date_3', 'vaccine_date_2', 'vaccine_date_1']].bfill(axis=1).iloc[:, 0]
    data['most_recent_vaccine'] = (REF_DATE - pd.to_datetime(most_recent_vaccine, format='%Y-%m-%d')).dt.days.fillna(700)

    most_recent_infection = df[['pcr_pos_date_2', 'pcr_pos_date_1']].bfill(axis=1).iloc[:, 0]
    data['most_recent_infection'] = (REF_DATE - pd.to_datetime(most_recent_infection, format='%Y-%m-%d')).dt.days.fillna(700)

    most_recent_contact = df[['pcr_pos_date_2', 'pcr_pos_date_1', 'vaccine_date_4', 'vaccine_date_3', 'vaccine_date_2', 'vaccine_date_1']].bfill(axis=1).iloc[:, 0]
    data['most_recent_contact'] = (REF_DATE - pd.to_datetime(most_recent_contact, format='%Y-%m-%d')).dt.days.fillna(700)

    data['pcr_pos_date'] = (REF_DATE - pd.to_datetime(df['pcr_pos_date_1'], format='%Y-%m-%d')).dt.days.fillna(-100)
    data['age'] = scale(df['age'])
    data['bmi'] = scale(df['bmi']).fillna(df['bmi'].mean())
    data['delta_t'] = (pd.to_datetime(df['pcr_pos_date_1'], format='%Y-%m-%d') - pd.to_datetime(df['vaccine_date_1'], format='%Y-%m-%d')).dt.days.fillna(500)
    data['comorbidity'] = df['comorbidity'].map({df['comorbidity'].unique()[0]: 0, df['comorbidity'].unique()[1]: 1})
    data['first_exposure_date'] = (REF_DATE - pd.to_datetime(df['first_exposure_date'], format='%Y-%m-%d')).dt.days.fillna(500)
    data['num_vaccines'] = df[['vaccine_date_1', 'vaccine_date_2', 'vaccine_date_3', 'vaccine_date_4']].notna().sum(axis=1)
    data['ab_chuv_igg_s_logratio'] = scale(df['ab_chuv_igg_s_logratio'])
    data['ab_chuv_igg_n_logratio'] = scale(df['ab_chuv_igg_n_logratio'])
    data['ab_chuv_iga_logratio'] = scale(df['ab_chuv_iga_logratio'])
    data['first_exposure'] = df['first_exposure'].map({df['first_exposure'].unique()[0]: 0, df['first_exposure'].unique()[1]: 1}).fillna(0.5)
    data['prior_exposure'] = df['prior_exposure'].map({df['prior_exposure'].unique()[0]: 0, df['prior_exposure'].unique()[1]: 1}).fillna(0.5)
    data['last_antibody_before_omicron_igg_n_logratio'] = scale(df['last_antibody_before_omicron_igg_n_logratio'])
    return data


def create_features_agnostic(df):
    """
    Preprocess the COVID-19 data by converting categorical features to numerical,
    handling missing values, and creating additional features.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Processed data ready for modeling
    """
    data = pd.get_dummies(df, columns=['timepoint'], dtype=int)
    data['study'] = data['study'].map({data['study'].unique()[0]: 0, data['study'].unique()[1]: 1})
    data = pd.get_dummies(data, columns=['pop_sample'], dtype=int)
    
    # Process dates
    data['index_date'] = (REF_DATE - pd.to_datetime(data['index_date'], format='%Y-%m-%d')).dt.days
    
    # Process demographic information
    data['age'] = scale(data['age'])
    data['sex'] = data['sex'].map({data['sex'].unique()[0]: 0, data['sex'].unique()[1]: 1})
    data['bmi'] = scale(data['bmi']).fillna(data['bmi'].mean())
    data = pd.get_dummies(data, columns=['smoking'], dtype=int)
    
    # Process medical conditions
    data['comorbidity'] = data['comorbidity'].map({data['comorbidity'].unique()[0]: 0, data['comorbidity'].unique()[1]: 1})
    
    # Process various medical conditions with missing value handling
    for condition in ['hypertension', 'diabetes', 'cvd', 'respiratory', 'ckd', 'cancer', 'immune_supp']:
        data[condition] = data[condition].map({data[condition].unique()[0]: 0, data[condition].unique()[1]: 1})
        data[condition] = data[condition].fillna(data[condition].mean())
    
    # Process socioeconomic factors
    data = pd.get_dummies(data, columns=['income_3l'], dummy_na=0, dtype=int)
    data = pd.get_dummies(data, columns=['employment_4l'], dtype=int)
    data = pd.get_dummies(data, columns=['education_4l'], prefix='education', dummy_na=0, dtype=int)
    data['nationality'] = data['nationality'].map({'Non-Swiss': 0, 'Swiss': 1})
    data['summary_bl_behaviour'] = scale(data['summary_bl_behaviour'])
    
    # Process symptoms
    data['symp_init'] = data['symp_init'].map({"No": 0, "Yes": 1})
    data['symp_init'] = data['symp_init'].fillna(data['symp_init'].mean())
    data = pd.get_dummies(data, columns=['symp_count_init_3l'], dummy_na=0, dtype=int)
    data['symp_sev_init_3l'] = data['symp_sev_init_3l'].map({"Mild to moderate": 0, "Severe to very severe": 1})
    data['symp_sev_init_3l'] = data['symp_sev_init_3l'].fillna(data['symp_sev_init_3l'].mean())
    
    # Process hospitalization
    data['hosp_2wks'] = data['hosp_2wks'].map({"No": 0, "Yes": 1})
    data['hosp_2wks'] = data['hosp_2wks'].fillna(data['hosp_2wks'].mean())
    data['icu_2wks'] = data['icu_2wks'].map({"No": 0, "Yes": 1})
    data['icu_2wks'] = data['icu_2wks'].fillna(data['icu_2wks'].mean())
    
    # Process serological status
    data['seropos_at_bl'] = data['seropos_at_bl'].map({"No": 0, "Yes": 1})
    data['seropos_at_bl'] = data['seropos_at_bl'].fillna(data['seropos_at_bl'].mean())
    data['prior_pos_pcr'] = data['prior_pos_pcr'].map({"No": 0, "Yes": 1})
    data['prior_pos_pcr'] = data['prior_pos_pcr'].fillna(data['prior_pos_pcr'].mean())
    data['prior_exposure'] = data['prior_exposure'].map({"No": 0, "Yes": 1})
    data['prior_exposure'] = data['prior_exposure'].fillna(data['prior_exposure'].mean())
    
    # Process exposure and vaccine data
    data['first_exposure_date'] = (REF_DATE - pd.to_datetime(data['first_exposure_date'], format='%Y-%m-%d')).dt.days
    data['first_exposure_date'] = data['first_exposure_date'].fillna(data['first_exposure_date'].mean())
    data['first_exposure'] = data['first_exposure'].map(
        {data['first_exposure'].unique()[0]: 0, data['first_exposure'].unique()[1]: 1}).fillna(0.5)
    
    # Process vaccine information
    data = pd.get_dummies(data, columns=['vaccine_type_1'], dummy_na=0, dtype=int)
    data = pd.get_dummies(data, columns=['vaccine_type_2'], dummy_na=0, dtype=int)
    data = pd.get_dummies(data, columns=['vaccine_type_3'], dummy_na=0, dtype=int)
    data = pd.get_dummies(data, columns=['vaccine_type_4'], dummy_na=0, dtype=int)
    
    # Process PCR test data
    data['pcr_pos_date_1'] = (REF_DATE - pd.to_datetime(data['pcr_pos_date_1'], format='%Y-%m-%d')).dt.days
    data['pcr_pos_date_1'] = data['pcr_pos_date_1'].fillna(data['pcr_pos_date_1'].mean())
    data['pcr_pos_date_2'] = (REF_DATE - pd.to_datetime(data['pcr_pos_date_2'], format='%Y-%m-%d')).dt.days
    data['pcr_pos_date_2'] = data['pcr_pos_date_2'].fillna(data['pcr_pos_date_2'].mean())
    data = pd.get_dummies(data, columns=['pcr_pos_sev_1'], dummy_na=0, dtype=int)
    
    # Process antibody data
    data['ab_chuv_iga_ratio'] = scale(data['ab_chuv_iga_ratio'])
    data['ab_chuv_iga_result'] = data['ab_chuv_iga_result'].map(
        {data['ab_chuv_iga_result'].unique()[0]: 0, data['ab_chuv_iga_result'].unique()[1]: 1})
    data['ab_chuv_igg_s_ratio'] = scale(data['ab_chuv_igg_s_ratio'])
    data['ab_chuv_igg_s_result'] = data['ab_chuv_igg_s_result'].map(
        {data['ab_chuv_igg_s_result'].unique()[0]: 0, data['ab_chuv_igg_s_result'].unique()[1]: 1})
    data['ab_chuv_igg_n_ratio'] = scale(data['ab_chuv_igg_n_ratio'])
    data['ab_chuv_igg_n_result'] = data['ab_chuv_igg_n_result'].map(
        {data['ab_chuv_igg_n_result'].unique()[0]: 0, data['ab_chuv_igg_n_result'].unique()[1]: 1})
    data['ab_chuv_iga_logratio'] = scale(data['ab_chuv_iga_logratio'])
    data['ab_chuv_igg_s_logratio'] = scale(data['ab_chuv_igg_s_logratio'])
    data['ab_chuv_igg_n_logratio'] = scale(data['ab_chuv_igg_n_logratio'])
    
    # Drop the ratio columns
    data.drop(columns=['ab_chuv_iga_ratio', 'ab_chuv_igg_s_ratio', 'ab_chuv_igg_n_ratio'], inplace=True)
    
    data['last_antibody_before_omicron_iga_logratio'] = scale(data['last_antibody_before_omicron_iga_logratio'])
    data['last_antibody_before_omicron_igg_n_logratio'] = scale(data['last_antibody_before_omicron_igg_n_logratio'])
    data['last_antibody_before_omicron_igg_s_logratio'] = scale(data['last_antibody_before_omicron_igg_s_logratio'])
    
    # Process additional data
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', inplace=True, axis=1)
        
    for vac_num in range(1, 5):
        data[f'vaccine_date_{vac_num}'] = (REF_DATE - pd.to_datetime(data[f'vaccine_date_{vac_num}'], 
                                                                     format='%Y-%m-%d')).dt.days
        data[f'vaccine_date_{vac_num}'] = data[f'vaccine_date_{vac_num}'].fillna(-100)
    
    # Fill missing behavior data
    data['prior_hyg'] = data['prior_hyg'].fillna(0.)
    data['prior_dist'] = data['prior_hyg'].fillna(0.)
    data['prior_mask_mand'] = data['prior_hyg'].fillna(0.)
    
    # Ensure consistency with original df
    data['study'] = df['study'].map({df['study'].unique()[0]: 0, df['study'].unique()[1]: 1})
    data['final_outcome_amp'] = df['final_outcome_amp']
    data['index_date'] = (REF_DATE - pd.to_datetime(df['index_date'], format='%Y-%m-%d')).dt.days
    
    # Create derived features
    # Most recent vaccine
    most_recent_vaccine = df[['vaccine_date_4', 'vaccine_date_3', 'vaccine_date_2', 'vaccine_date_1']].bfill(axis=1).iloc[:, 0]
    most_recent_vaccine = (REF_DATE - pd.to_datetime(most_recent_vaccine, format='%Y-%m-%d')).dt.days.fillna(700)
    data['most_recent_vaccine'] = most_recent_vaccine
    
    # Most recent infection
    most_recent_infection = df[['pcr_pos_date_2', 'pcr_pos_date_1']].bfill(axis=1).iloc[:, 0]
    most_recent_infection = (REF_DATE - pd.to_datetime(most_recent_infection, format='%Y-%m-%d')).dt.days.fillna(700)
    data['most_recent_infection'] = most_recent_infection
    
    # Most recent contact (vaccine or infection)
    most_recent_contact = df[['pcr_pos_date_2', 'pcr_pos_date_1', 
                             'vaccine_date_4', 'vaccine_date_3', 
                             'vaccine_date_2', 'vaccine_date_1']].bfill(axis=1).iloc[:, 0]
    most_recent_contact = (REF_DATE - pd.to_datetime(most_recent_contact, format='%Y-%m-%d')).dt.days.fillna(700)
    data['most_recent_contact'] = most_recent_contact
    
    return data

def create_features_sequence(df_main, df_sequence):
    """Creates and combines features for the 'sequence' analysis."""
    # Main features
    data = pd.DataFrame(index=df_main.index)
    data['study'] = df_main['study'].map({df_main['study'].unique()[0]: 0, df_main['study'].unique()[1]: 1})
    data['final_outcome_amp'] = df_main['final_outcome_amp']
    #data['index_date'] = (REF_DATE - pd.to_datetime(df_main['index_date'], format='%Y-%m-%d')).dt.days
    data['pcr_pos_date'] = (REF_DATE - pd.to_datetime(df_main['pcr_pos_date_1'], format='%Y-%m-%d')).dt.days.fillna(-100)
    data['age'] = scale(df_main['age'])
    data['bmi'] = scale(df_main['bmi']).fillna(df_main['bmi'].mean())
    data['comorbidity'] = df_main['comorbidity'].map({df_main['comorbidity'].unique()[0]: 0, df_main['comorbidity'].unique()[1]: 1})
    data['first_exposure_date'] = (REF_DATE - pd.to_datetime(df_main['first_exposure_date'], format='%Y-%m-%d')).dt.days.fillna(500)
    data['num_vaccines'] = df_main[['vaccine_date_1', 'vaccine_date_2', 'vaccine_date_3', 'vaccine_date_4']].notna().sum(axis=1)
    data['ab_chuv_igg_s_logratio'] = scale(df_main['ab_chuv_igg_s_logratio'])
    data['ab_chuv_igg_n_logratio'] = scale(df_main['ab_chuv_igg_n_logratio'])
    data['ab_chuv_iga_logratio'] = scale(df_main['ab_chuv_iga_logratio'])
    data['first_exposure'] = df_main['first_exposure'].map({df_main['first_exposure'].unique()[0]: 0, df_main['first_exposure'].unique()[1]: 1}).fillna(0.5)
    data['prior_exposure'] = df_main['prior_exposure'].map({df_main['prior_exposure'].unique()[0]: 0, df_main['prior_exposure'].unique()[1]: 1}).fillna(0.5)
    data['last_antibody_before_omicron_igg_n_logratio'] = scale(df_main['last_antibody_before_omicron_igg_n_logratio'])

    # Sequence-specific features
    data_seq = pd.DataFrame(index=df_sequence.index)
    data_seq = pd.get_dummies(df_sequence, columns=['sequence_cat', 'sequence'], dtype=int)
    data_seq['days_between_lastexp'] = scale(df_sequence['days_between_lastexp'])
    data_seq['days_betweentp'] = scale(df_sequence['days_betweentp'])
    data_seq['behaviour_cat_2l_v2'] = df_sequence['behaviour_cat_2l_v2'].map({df_sequence['behaviour_cat_2l_v2'].unique()[0]: 0, df_sequence['behaviour_cat_2l_v2'].unique()[1]: 1})
    
    # Join dataframes
    return data.join(data_seq, on='record_id', how='left')


# --- MODEL TRAINING & EVALUATION ---

def train_and_evaluate(X, y, data_mode, config_name=None):
    """Defines models, trains them using GridSearchCV, and evaluates performance."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'param_grid': {'bootstrap': [True, False], 'n_estimators': [20, 100, 200, 500], 'max_depth': [None, 10, 20, 40], 'min_samples_split': [5, 7, 10, 12, 20, 50], 'min_samples_leaf': [2, 5, 8], 'class_weight': ['balanced', None]}
        },
         'XGBoost': {
            'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', tree_method="hist", use_label_encoder=False, gpu_id=-1),
            'param_grid': {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100, 200, 500], 'max_depth': [3, 5], 'reg_lambda': [0, 0.1, 1], 'min_child_weight': [1, 3, 5, 10], 'scale_pos_weight': [1, (len(y) - sum(y)) / sum(y)]}
        }, 
         'SVM': {
            'model': SVC(probability=True, random_state=RANDOM_STATE),
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced', None]
            }
        }, 
        'LogisticRegression': {
        'model': LogisticRegression(random_state=42, solver='liblinear'),
        'param_grid': {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1, 10],
            'class_weight': ['balanced', None]
        }
    },
        'LinearDiscriminantAnalysis': {
            'model': LinearDiscriminantAnalysis(),
            'param_grid': [
                {'solver': ['svd'], 'tol': [1e-4, 1e-3, 1e-2]},  # no shrinkage for svd
                {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 0.9], 'tol': [1e-4, 1e-3, 1e-2]}
            ]
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'param_grid': {'max_depth': [2, 5, 15, 20], 'min_samples_leaf': [2, 25, 50]}
        },
        'LinearDiscriminantAnalysis': {
            'model': LinearDiscriminantAnalysis(),
            'param_grid': {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 0.9], 'tol': [1e-4, 1e-3, 1e-2]}
        }
    }
    models.pop("XGBoost")
    models.pop("SVM")

    plt.figure(figsize=(10, 8))
    best_models = {}
    accuracy_scores = {}
    auc_scores = {}
    balanced_accuracy_scores = {}
    f1_scores = {}
    sensitivity_scores = {}
    specificity_scores = {}
    feature_importances = {}
    optimal_parameters = {}
    aucs_per_model_per_split = {}

    for model_name, config in models.items():
        print(f"--- Training {model_name} ---")
        tprs, aucs, accuracies, balanced_accuracies = [], [], [], []
        f1s, sensitivities, specificities = [], [], []
        mean_fpr = np.linspace(0, 1, 100)
        optimal_parameters[model_name] = []

        for split in range(N_SPLITS):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE + split, stratify=y)
            grid_search = GridSearchCV(estimator=config['model'], param_grid=config['param_grid'], scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            optimal_parameters[model_name].append(grid_search.best_params_)
            best_models[model_name] = best_model
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc_score(y_test, y_proba))
            accuracies.append(accuracy_score(y_test, y_pred))
            balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
            # F1, Sensitivity, Specificity
            from sklearn.metrics import f1_score, confusion_matrix
            f1s.append(f1_score(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2,2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                sensitivity = 0.0
                specificity = 0.0
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            if model_name == 'RandomForest' and split == N_SPLITS - 1:
                feature_importances[model_name] = best_model.feature_importances_

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2)
        accuracy_scores[model_name] = (np.mean(accuracies), np.std(accuracies))
        auc_scores[model_name] = (mean_auc, std_auc)
        balanced_accuracy_scores[model_name] = (np.mean(balanced_accuracies), np.std(balanced_accuracies))
        f1_scores[model_name] = (np.mean(f1s), np.std(f1s))
        sensitivity_scores[model_name] = (np.mean(sensitivities), np.std(sensitivities))
        specificity_scores[model_name] = (np.mean(specificities), np.std(specificities))
        aucs_per_model_per_split[model_name] = aucs

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    title_suffix = f' ({config_name})' if config_name else ''
    plt.title(f'ROC Curves (Mode: {data_mode}{title_suffix}, {N_SPLITS} Random Splits)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    save_suffix = f'_{config_name}' if config_name else ''
    plt.savefig(f'roc_curves_{data_mode}{save_suffix}_{N_SPLITS}_splits.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n=== Optimal Hyperparameters (Most Common Across Splits) ===")
    for model_name, params_list in optimal_parameters.items():
        print(f"\n** {model_name} **")
        # Collect all unique parameter names across all splits
        all_param_names = set()
        for p in params_list:
            all_param_names.update(p.keys())
        param_summary = {}
        for param_name in all_param_names:
            values = [p.get(param_name) for p in params_list if param_name in p]
            if values:
                param_summary[param_name] = Counter(values).most_common(1)[0][0]
        for param, value in param_summary.items():
            print(f"{param}: {value}")

    print("\n=== Mean Accuracy, AUC, F1, Sensitivity, Specificity (All Splits) ===")
    for model_name in models.keys():
        acc_mean, acc_std = accuracy_scores[model_name]
        bal_acc_mean, bal_acc_std = balanced_accuracy_scores[model_name]
        auc_mean, auc_std = auc_scores[model_name]
        f1_mean, f1_std = f1_scores[model_name]
        sens_mean, sens_std = sensitivity_scores[model_name]
        spec_mean, spec_std = specificity_scores[model_name]
        print(f"\n** {model_name} **")
        print(f"Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"Balanced Accuracy: {bal_acc_mean:.3f} ± {bal_acc_std:.3f}")
        print(f"AUC: {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"F1 Score: {f1_mean:.3f} ± {f1_std:.3f}")
        print(f"Sensitivity (Recall): {sens_mean:.3f} ± {sens_std:.3f}")
        print(f"Specificity: {spec_mean:.3f} ± {spec_std:.3f}")

    if 'RandomForest' in feature_importances:
        print("\n=== Feature Importances for RandomForest (Last Split) ===")
        feature_importance_dict = dict(zip(X.columns, feature_importances['RandomForest']))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")

    return aucs_per_model_per_split


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Load and filter base data once
    all_data = pd.read_csv(DATA_FILE)
    raw_data = filter_data_date(all_data).dropna(axis=1, how='all')
    raw_data.to_csv(join(DATA_DIR,'filtered_data.csv'))
    raw_data.pop('visit_date')
    raw_data.set_index('record_id', inplace=True)
    raw_data.columns = raw_data.columns.astype(str).str.replace(r'[\\[\\]<>]', '', regex=True)

    print(f"\n{'='*50} TESTING AGNOSTIC MODE WITH DIFFERENT FEATURE SELECTION FLAGS {'='*50}\n")

    all_results = {}
    all_metrics = {}
    
    # Test each configuration
    for config_name, config in TEST_CONFIGS.items():
        print(f"\n{'='*20} TESTING CONFIGURATION: {config_name.upper()} {'='*20}")
        print(f"Configuration: {config}")
        
        # Create agnostic features
        data = create_features_agnostic(raw_data)
        data.dropna(inplace=True)
        X = data.drop(TARGET_COL, axis=1)
        y = data[TARGET_COL]
        
        print(f"\nShape before any filtering: {X.shape}")
        
        # Apply feature selection based on configuration
        if config['FILTER_MI']:
            X = filter_by_mutual_information(X, y, k=K_MI)
        if config['FILTER_CORRELATION']:
            X = filter_by_correlation(X, threshold=CORR_THRESHOLD)
        if config['FILTER_RF']:
            X = filter_by_rf_importance(X, y, k=K_RF)
        if config['APPLY_PCA']:
            X = apply_pca(X, n_components=PCA_COMPONENTS)
        
        # Re-align target with filtered features
        y = y.loc[X.index]
        print(f"Shape after filtering: {X.shape}")
        
        # Clean column names for models like XGBoost
        X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X.columns]
        
        print(f"\nFinal shape of features X: {X.shape}")
        print(f"Final shape of target y: {y.shape}")
        print("\nTarget distribution:\n", y.value_counts(normalize=True))
        
        # Run training and evaluation pipeline
        results = train_and_evaluate(X, y, 'agnostic', config_name)
        all_results[config_name] = results
        
        # Store metrics for comparison
        config_metrics = {}
        for model_name, aucs in results.items():
            config_metrics[model_name] = {
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs)
            }
        all_metrics[config_name] = config_metrics

    # --- COMPARISON OF CONFIGURATIONS ---
    print(f"\n{'='*50} CONFIGURATION COMPARISON {'='*50}")
    
    # Find best configuration for each model
    best_configs = {}
    for model_name in all_results[list(TEST_CONFIGS.keys())[0]].keys():
        best_config = None
        best_auc = -1
        
        print(f"\n=== {model_name} ===")
        print(f"{'Configuration':<20} {'Mean AUC':<12} {'Std AUC':<12}")
        print("-" * 50)
        
        for config_name in TEST_CONFIGS.keys():
            if model_name in all_metrics[config_name]:
                mean_auc = all_metrics[config_name][model_name]['mean_auc']
                std_auc = all_metrics[config_name][model_name]['std_auc']
                print(f"{config_name:<20} {mean_auc:<12.4f} {std_auc:<12.4f}")
                
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_config = config_name
        
        best_configs[model_name] = best_config
        print(f"BEST: {best_config} (AUC: {best_auc:.4f})")
    
    # Overall summary
    print(f"\n{'='*20} OVERALL BEST CONFIGURATIONS {'='*20}")
    config_counts = {}
    for model_name, best_config in best_configs.items():
        if best_config not in config_counts:
            config_counts[best_config] = []
        config_counts[best_config].append(model_name)
        print(f"{model_name}: {best_config}")
    
    print(f"\n{'='*20} CONFIGURATION SUMMARY {'='*20}")
    for config_name, models in config_counts.items():
        print(f"{config_name}: Best for {len(models)} models ({', '.join(models)})")
    
    # Find most frequently best configuration
    most_frequent_config = max(config_counts.keys(), key=lambda x: len(config_counts[x]))
    print(f"\nMost frequently best configuration: {most_frequent_config}")
    print(f"Configuration details: {TEST_CONFIGS[most_frequent_config]}")
    
    print(f"\n{'='*50} ANALYSIS COMPLETE {'='*50}")
    print(f"Generated ROC curve plots for each configuration:")
    for config_name in TEST_CONFIGS.keys():
        print(f"  - roc_curves_agnostic_{config_name}_{N_SPLITS}_splits.png")