import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


SCALE = True

# Flags for 'agnostic' mode
APPLY_PCA = False
FILTER_MI = False#True
FILTER_CORRELATION = False
FILTER_RF = False
K_MI = 15  # Number of features to select with Mutual Information
K_RF = 15  # Number of features to select with RF Importance
CORR_THRESHOLD = 0.9
PCA_COMPONENTS = 0.95

N_SPLITS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

DATA_DIR = 'data/inputs'
DATA_FILE = join(DATA_DIR, 'ZSAC_ZVAC_Collaboration_data_v3_20241121.csv')
DATA_FILE_SEQUENCE = join(DATA_DIR, 'ZSAC_ZVAC_Collaboration_sequence_data_20250124.csv')

REF_DATE = pd.to_datetime('2022-01-01')
TARGET_COL = 'final_outcome_amp'
ROUND_LABELS = {
    'agnostic': 'Round 1.1',
    'background': 'Round 2.0',
    'sequence': 'Round 2.1'
}



def find_optimal_threshold_f1(y_true, y_proba):
    """Find optimal threshold that maximizes F1 score."""
    thresholds = np.arange(0.1, 1.0, 0.01)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]

def find_optimal_threshold_youden(y_true, y_proba):
    """Find optimal threshold using Youden's J statistic (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx], youden_j[optimal_idx]

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
    
    data['final_outcome_amp'] = df['final_outcome_amp']
    data['index_date'] = (REF_DATE - pd.to_datetime(df['index_date'], format='%Y-%m-%d')).dt.days
    return data

def create_features_sequence(df_main, df_sequence):
    """Creates and combines features for the 'sequence' analysis."""
    data = create_features_background(df_main)

    data_seq = pd.DataFrame(index=df_sequence.index)
    data_seq = pd.get_dummies(df_sequence, columns=['sequence_cat', 'sequence'], dtype=int)
    data_seq['days_between_lastexp'] = scale(df_sequence['days_between_lastexp'])
    data_seq['days_betweentp'] = scale(df_sequence['days_betweentp'])
    data_seq['behaviour_cat_2l_v2'] = df_sequence['behaviour_cat_2l_v2'].map({df_sequence['behaviour_cat_2l_v2'].unique()[0]: 0, df_sequence['behaviour_cat_2l_v2'].unique()[1]: 1})
    
    return data.join(data_seq, on='record_id', how='left')


# --- MODEL TRAINING & EVALUATION ---

def train_and_evaluate(X, y, data_mode):
    """Defines models, trains them using GridSearchCV, and evaluates performance.

    Returns
    -------
    aucs_per_model_per_split : dict
        Mapping of model name -> list of AUROC values across splits.
    optimal_params_summary : dict
        Mapping of model name -> dict of most common/best hyperparameters across splits.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'search_spaces': {
                'n_estimators': Integer(50, 1000),
                'max_depth': Integer(3, 60),
                'min_samples_split': Integer(2, 50),
                'min_samples_leaf': Integer(1, 20),
                'max_features': Categorical([None, 'sqrt', 'log2']),
                'bootstrap': Categorical([True, False]),
                'criterion': Categorical(['gini', 'entropy']),
                'class_weight': Categorical(['balanced', None])
            }
        },
        #'SVM': {
        #    'model': SVC(probability=True, random_state=RANDOM_STATE),
        #    'search_spaces': {
        #        'C': Real(1e-3, 1e3, prior='log-uniform'),
        #        'kernel': Categorical(['linear', 'rbf', 'poly', 'sigmoid']),
        #        'degree': Integer(2, 5),
        #        'gamma': Categorical(['scale', 'auto']),
        #        'class_weight': Categorical(['balanced', None])
        #    }
        #},
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, solver='liblinear', max_iter=2000),
            'search_spaces': [
            {
                'solver': Categorical(['liblinear']),
                'penalty': Categorical(['l1', 'l2']),
                'C': Real(1e-3, 1e3, prior='log-uniform'),
                'class_weight': Categorical(['balanced', None])
            },
            {
                'solver': Categorical(['saga']),
                'penalty': Categorical(['l1', 'l2']),
                'C': Real(1e-3, 1e3, prior='log-uniform'),
                'class_weight': Categorical(['balanced', None])
            }
            ]
        },
        'LinearDiscriminantAnalysis': {
            'model': LinearDiscriminantAnalysis(),
            'search_spaces': {
            'solver': Categorical(['svd']),
            'tol': Real(1e-6, 1e-2, prior='log-uniform')
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'search_spaces': {
            'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
            'n_estimators': Integer(50, 1000),
            'max_depth': Integer(2, 30),
            'min_samples_leaf': Integer(1, 60),
            'subsample': Real(0.5, 1.0),
            'max_features': Categorical([None, 'sqrt', 'log2'])
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', tree_method='hist', predictor='cpu_predictor', verbosity=0),
            'bayes_n_jobs': 1,
            'search_spaces': {
                'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
                'n_estimators': Integer(50, 1000),
                'max_depth': Integer(3, 12),
                'reg_lambda': Real(0.0, 10.0),
                'min_child_weight': Integer(1, 10),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.3, 1.0)
            }
        }
    }
    models = {"RandomForest": models["RandomForest"]}

    plt.figure(figsize=(10, 8))
    def format_mean_ci(mean, std, n):
        if n <= 0:
            return f"{mean:.3f}"
        se = std / np.sqrt(n)
        z = 1.96
        lower = mean - z * se
        upper = mean + z * se
        return f"{mean:.3f} (95% CI: {lower:.3f} - {upper:.3f})"
    best_models = {}
    fitted_models_per_split = {name: [] for name in models.keys()}
    fitted_feature_names_per_split = {name: [] for name in models.keys()}
    accuracy_scores = {}
    auc_scores = {}
    balanced_accuracy_scores = {}
    f1_scores = {}
    sensitivity_scores = {}
    specificity_scores = {}
    precision_scores = {}
    aupr_scores = {}
    f1_optimized_scores = {}
    precision_optimized_scores = {}
    recall_optimized_scores = {}
    balanced_accuracy_optimized_scores = {}
    feature_importances = {}
    optimal_parameters = {}
    aucs_per_model_per_split = {}

    for model_name, config in models.items():
        print(f"--- Training {model_name} ---")
        tprs, aucs, accuracies, balanced_accuracies = [], [], [], []
        f1s, sensitivities, specificities, precisions, auprs = [], [], [], [], []
        f1s_optimized, precisions_optimized, recalls_optimized, balanced_accuracies_optimized = [], [], [], []
        mean_fpr = np.linspace(0, 1, 100)
        optimal_parameters[model_name] = []

        for split in range(N_SPLITS):
            print(f" Split {split + 1}/{N_SPLITS}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE + split, stratify=y)
            bayes_search = BayesSearchCV(
                estimator=config['model'],
                search_spaces=config['search_spaces'],
                scoring='roc_auc',
                cv=cv,
                # Allow per-model override (e.g. XGBoost -> 1) via 'bayes_n_jobs'
                n_jobs=config.get('bayes_n_jobs', -1),
                n_iter=32,
                random_state=RANDOM_STATE + split,
                verbose=0
            )
            # Special-case handling: if using LDA and X_train is rank-deficient
            # (rank < n_features), LDA's covariance computations may fail during
            # BayesSearchCV. In that case, skip the expensive Bayes search and
            # fit the base estimator directly for this split.
            if model_name == 'LinearDiscriminantAnalysis':
                try:
                    rank = np.linalg.matrix_rank(X_train.values)
                except Exception:
                    rank = None
                if rank is not None and rank < X_train.shape[1]:
                    print(f"Info: Skipping BayesSearchCV for LDA on split {split+1} due to rank-deficient X_train (rank={rank}, n_features={X_train.shape[1]}). Fitting base estimator.")
                    base_model = config['model']
                    base_model.fit(X_train, y_train)
                    best_model = base_model
                    optimal_parameters[model_name].append({})
                    best_models[model_name] = best_model
                    # proceed to evaluation
                    pass
                else:
                    try:
                        bayes_search.fit(X_train, y_train)
                        best_model = bayes_search.best_estimator_
                        optimal_parameters[model_name].append(bayes_search.best_params_)
                        best_models[model_name] = best_model
                    except Exception as e:
                        # Catch LinAlgError and any other issues from invalid parameter combinations
                        print(f"Warning: BayesSearchCV failed for {model_name} on split {split+1} with error: {e}")
                        print("Falling back to fitting the base estimator with default parameters for this split.")
                        try:
                            base_model = config['model']
                            base_model.fit(X_train, y_train)
                            best_model = base_model
                            optimal_parameters[model_name].append({})
                            best_models[model_name] = best_model
                        except Exception as e2:
                            print(f"Error: fallback fit also failed for {model_name} on split {split+1}: {e2}")
                            # Re-raise to avoid silent failures
                            raise
            else:
                try:
                    bayes_search.fit(X_train, y_train)
                    best_model = bayes_search.best_estimator_
                    optimal_parameters[model_name].append(bayes_search.best_params_)
                    best_models[model_name] = best_model
                except Exception as e:
                    # Catch LinAlgError and any other issues from invalid parameter combinations
                    print(f"Warning: BayesSearchCV failed for {model_name} on split {split+1} with error: {e}")
                    print("Falling back to fitting the base estimator with default parameters for this split.")
                    try:
                        base_model = config['model']
                        base_model.fit(X_train, y_train)
                        best_model = base_model
                        optimal_parameters[model_name].append({})
                        best_models[model_name] = best_model
                    except Exception as e2:
                        print(f"Error: fallback fit also failed for {model_name} on split {split+1}: {e2}")
                        # Re-raise to avoid silent failures
                        raise
            # Append the actual fitted model and the feature names used for this split
            try:
                fitted_models_per_split[model_name].append(best_model)
                fitted_feature_names_per_split[model_name].append(list(X_train.columns))
            except Exception:
                # Ensure we always have placeholders to keep list lengths consistent
                fitted_models_per_split[model_name].append(None)
                fitted_feature_names_per_split[model_name].append(None)
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Standard metrics with default threshold (0.5)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc_score(y_test, y_proba))
            accuracies.append(accuracy_score(y_test, y_pred))
            balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
            
            # Calculate AUPR (Area Under Precision-Recall Curve)
            aupr = average_precision_score(y_test, y_proba)
            auprs.append(aupr)
            
            # Standard F1, Precision, Sensitivity, Specificity with default threshold
            f1s.append(f1_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            
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
            
            # Optimized threshold metrics
            # Find optimal threshold using F1 score
            optimal_threshold_f1, _ = find_optimal_threshold_f1(y_test, y_proba)
            y_pred_optimized_f1 = (y_proba >= optimal_threshold_f1).astype(int)
            
            f1s_optimized.append(f1_score(y_test, y_pred_optimized_f1))
            precisions_optimized.append(precision_score(y_test, y_pred_optimized_f1, zero_division=0))
            recalls_optimized.append(recall_score(y_test, y_pred_optimized_f1, zero_division=0))
            balanced_accuracies_optimized.append(balanced_accuracy_score(y_test, y_pred_optimized_f1))
            
            if model_name == 'RandomForest' and split == N_SPLITS - 1:
                feature_importances[model_name] = best_model.feature_importances_
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
        
        # Store all metrics
        accuracy_scores[model_name] = (np.mean(accuracies), np.std(accuracies))
        auc_scores[model_name] = (mean_auc, std_auc)
        balanced_accuracy_scores[model_name] = (np.mean(balanced_accuracies), np.std(balanced_accuracies))
        f1_scores[model_name] = (np.mean(f1s), np.std(f1s))
        sensitivity_scores[model_name] = (np.mean(sensitivities), np.std(sensitivities))
        specificity_scores[model_name] = (np.mean(specificities), np.std(specificities))
        precision_scores[model_name] = (np.mean(precisions), np.std(precisions))
        aupr_scores[model_name] = (np.mean(auprs), np.std(auprs))
        
        # Store optimized threshold metrics
        f1_optimized_scores[model_name] = (np.mean(f1s_optimized), np.std(f1s_optimized))
        precision_optimized_scores[model_name] = (np.mean(precisions_optimized), np.std(precisions_optimized))
        recall_optimized_scores[model_name] = (np.mean(recalls_optimized), np.std(recalls_optimized))
        balanced_accuracy_optimized_scores[model_name] = (np.mean(balanced_accuracies_optimized), np.std(balanced_accuracies_optimized))
        
        aucs_per_model_per_split[model_name] = aucs

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    round_label = ROUND_LABELS.get(data_mode, data_mode)
    plt.title(f'ROC Curves ({round_label}, {N_SPLITS} Random Splits)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'roc_curves_{data_mode}_{N_SPLITS}_splits.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n=== Optimal Hyperparameters (Most Common Across Splits) ===")
    optimal_params_summary = {}
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
        optimal_params_summary[model_name] = param_summary
        for param, value in param_summary.items():
            print(f"{param}: {value}")

    print("\n=== Mean Accuracy, AUC, F1, Sensitivity, Specificity, Precision, AUPR (All Splits) ===")
    for model_name in models.keys():
        acc_mean, acc_std = accuracy_scores[model_name]
        bal_acc_mean, bal_acc_std = balanced_accuracy_scores[model_name]
        auc_mean, auc_std = auc_scores[model_name]
        f1_mean, f1_std = f1_scores[model_name]
        sens_mean, sens_std = sensitivity_scores[model_name]
        spec_mean, spec_std = specificity_scores[model_name]
        prec_mean, prec_std = precision_scores[model_name]
        aupr_mean, aupr_std = aupr_scores[model_name]
        n = len(aucs_per_model_per_split.get(model_name, [])) or N_SPLITS
        
        print(f"\n** {model_name} **")
        print(f"Accuracy: {format_mean_ci(acc_mean, acc_std, n)}")
        print(f"Balanced Accuracy: {format_mean_ci(bal_acc_mean, bal_acc_std, n)}")
        print(f"AUC (AUROC): {format_mean_ci(auc_mean, auc_std, n)}")
        print(f"AUPR (Average Precision): {format_mean_ci(aupr_mean, aupr_std, n)}")
        print(f"F1 Score: {format_mean_ci(f1_mean, f1_std, n)}")
        print(f"Precision: {format_mean_ci(prec_mean, prec_std, n)}")
        print(f"Sensitivity (Recall): {format_mean_ci(sens_mean, sens_std, n)}")
        print(f"Specificity: {format_mean_ci(spec_mean, spec_std, n)}")

    print("\n=== Optimized Threshold Metrics (F1-Optimized Threshold) ===")
    for model_name in models.keys():
        f1_opt_mean, f1_opt_std = f1_optimized_scores[model_name]
        prec_opt_mean, prec_opt_std = precision_optimized_scores[model_name]
        recall_opt_mean, recall_opt_std = recall_optimized_scores[model_name]
        bal_acc_opt_mean, bal_acc_opt_std = balanced_accuracy_optimized_scores[model_name]
        n = len(aucs_per_model_per_split.get(model_name, [])) or N_SPLITS
        
        print(f"\n** {model_name} (F1-Optimized Threshold) **")
        print(f"F1 Score (Optimized): {format_mean_ci(f1_opt_mean, f1_opt_std, n)}")
        print(f"Precision (Optimized): {format_mean_ci(prec_opt_mean, prec_opt_std, n)}")
        print(f"Recall (Optimized): {format_mean_ci(recall_opt_mean, recall_opt_std, n)}")
        print(f"Balanced Accuracy (Optimized): {format_mean_ci(bal_acc_opt_mean, bal_acc_opt_std, n)}")

    if 'RandomForest' in feature_importances:
        print("\n=== Feature Importances for RandomForest (Last Split) ===")
        feature_importance_dict = dict(zip(X.columns, feature_importances['RandomForest']))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")

    return aucs_per_model_per_split, optimal_params_summary, optimal_parameters, fitted_models_per_split, fitted_feature_names_per_split


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Load and filter base data once
    all_data = pd.read_csv(DATA_FILE)
    raw_data = filter_data_date(all_data).dropna(axis=1, how='all')
    raw_data.to_csv(join(DATA_DIR,'filtered_data.csv'))
    raw_data.pop('visit_date')
    raw_data.set_index('record_id', inplace=True)
    raw_data.columns = raw_data.columns.astype(str).str.replace(r'[\\[\\]<>]', '', regex=True)

    # Load sequence data once
    all_data_sequence = pd.read_csv(DATA_FILE_SEQUENCE)
    raw_data_sequence = all_data_sequence.dropna(axis=1, how='all')
    raw_data_sequence.set_index('record_id', inplace=True)
    raw_data_sequence.columns = raw_data_sequence.columns.astype(str).str.replace(r'[\\[\\]<>]', '', regex=True)

    all_results = {}
    all_optimal_params = {}
    all_best_models = {}
    all_best_features = {}
    for data_mode in ['agnostic', 'background', 'sequence']:
        print(f"\n{'='*20} RUNNING IN MODE: {data_mode.upper()} {'='*20}\n")

        # Create features based on selected mode
        if data_mode == 'background':
            data = create_features_background(raw_data)
        elif data_mode == 'agnostic':
            data = create_features_agnostic(raw_data)
        elif data_mode == 'sequence':
            data = create_features_sequence(raw_data, raw_data_sequence)

        # Post-processing and splitting
        data.dropna(inplace=True) # Drop rows with any NaNs after feature creation/joining
        X = data.drop(TARGET_COL, axis=1)
        y = data[TARGET_COL]

        print(f"\nShape before any filtering: {X.shape}")

        # Apply feature selection for 'agnostic' mode
        if data_mode == 'agnostic':
            if FILTER_MI:
                X = filter_by_mutual_information(X, y, k=K_MI)
            if FILTER_CORRELATION:
                X = filter_by_correlation(X, threshold=CORR_THRESHOLD)
            if FILTER_RF:
                X = filter_by_rf_importance(X, y, k=K_RF)
            if APPLY_PCA:
                X = apply_pca(X, n_components=PCA_COMPONENTS)
            y = y.loc[X.index]  # Re-align target with filtered features
            print(f"Shape after agnostic filtering: {X.shape}")

        # Clean column names for models like XGBoost
        X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X.columns]

        print(f"\nFinal shape of features X: {X.shape}")
        print(f"Final shape of target y: {y.shape}")
        print("\nTarget distribution:\n", y.value_counts(normalize=True))
        # Train & evaluate for this data mode and store results immediately
        results, optimal_params_summary, split_wise_params, fitted_models, fitted_feature_names = train_and_evaluate(X, y, data_mode)
        all_results[data_mode] = results
        all_optimal_params[data_mode] = optimal_params_summary
        # Store per-split params for exact reproduction when plotting ROC comparisons
        all_optimal_params.setdefault('split_wise', {})
        all_optimal_params['split_wise'][data_mode] = split_wise_params
        # Store the actual fitted models per split so ROC comparison uses the identical estimators
        all_best_models[data_mode] = fitted_models
        # Also store the trained feature names per split
        all_best_features[data_mode] = fitted_feature_names

    # --- BOOTSTRAP ANALYSIS ---
    background_aucs = all_results.get('background', {})
    agnostic_aucs = all_results.get('agnostic', {})
    sequence_aucs = all_results.get('sequence', {})

    if background_aucs and agnostic_aucs:
        print("\n=== AUROC Difference (Background - Agnostic) across splits ===")
        for model_name in background_aucs.keys():
            if model_name in agnostic_aucs:
                diffs = np.array(background_aucs[model_name]) - np.array(agnostic_aucs[model_name])
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                print(f"\n** {model_name} **")
                print(f"  Mean AUROC Difference: {mean_diff:.4f}")
                print(f"  Std Dev of Difference: {std_diff:.4f}")

                # Plot bar chart of differences per split
                plt.figure(figsize=(10, 6))
                splits = range(1, len(diffs) + 1)
                plt.bar(splits, diffs, color='skyblue', edgecolor='black')
                plt.title(f'AUROC Difference per Split for {model_name}\n(Round 2.0 - Round 1.1)')
                plt.xlabel('Split Number')
                plt.ylabel('AUROC Difference')
                plt.axhline(mean_diff, color='darkblue', linestyle='dashed', linewidth=2, label=f'Mean Diff: {mean_diff:.4f}')
                plt.xticks(splits)
                plt.legend()
                plt.grid(True, axis='y', alpha=0.3)
                plt.savefig(f'auroc_diff_barchart_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.close()

    # --- AUROC Difference (Sequence - Background) ---
    if background_aucs and sequence_aucs:
        print("\n=== AUROC Difference (Sequence - Background) across splits ===")
        for model_name in background_aucs.keys():
            if model_name in sequence_aucs:
                diffs = np.array(sequence_aucs[model_name]) - np.array(background_aucs[model_name])
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                print(f"\n** {model_name} **")
                print(f"  Mean AUROC Difference: {mean_diff:.4f}")
                print(f"  Std Dev of Difference: {std_diff:.4f}")

                # Plot bar chart of differences per split
                plt.figure(figsize=(10, 6))
                splits = range(1, len(diffs) + 1)
                plt.bar(splits, diffs, color='lightcoral', edgecolor='black')
                plt.title(f'AUROC Difference per Split for {model_name}\n(Sequence - Background)')
                plt.xlabel('Split Number')
                plt.ylabel('AUROC Difference')
                plt.axhline(mean_diff, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Diff: {mean_diff:.4f}')
                plt.xticks(splits)
                plt.legend()
                plt.grid(True, axis='y', alpha=0.3)
                plt.savefig(f'auroc_diff_seq_vs_bg_barchart_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
    # --- ROC COMPARISON ACROSS DATA TYPES FOR EACH MODEL ---
    # For each model, plot ROC curves for all three data types on the same figure
    print("\n=== Plotting ROC curve comparison for each model across data types ===")
    # Use round labels in the legends: Agnostic -> Round 1.1, Background -> Round 2.0, Sequence -> Round 2.1
    data_type_labels = {
        'background': f"{ROUND_LABELS.get('background')}",
        'agnostic': f"{ROUND_LABELS.get('agnostic')}",
        'sequence': f"{ROUND_LABELS.get('sequence')}"
    }
    colors = {'background': 'tab:blue', 'agnostic': 'tab:orange', 'sequence': 'tab:green'}
    # For ROC curves, we need to re-run the mean ROC calculation for each model/data_type
    for model_name in all_results['background'].keys():
        plt.figure(figsize=(10, 8))
        for data_mode in ['background', 'agnostic', 'sequence']:
            if model_name in all_results.get(data_mode, {}):
                # To get mean ROC, we need to re-run the splits for each data_mode/model_name
                # We'll reload the data and features for each data_mode
                # Use the same feature creation logic as above
                if data_mode == 'background':
                    data = create_features_background(raw_data)
                elif data_mode == 'agnostic':
                    data = create_features_agnostic(raw_data)
                elif data_mode == 'sequence':
                    data = create_features_sequence(raw_data, raw_data_sequence)
                data.dropna(inplace=True)
                X = data.drop(TARGET_COL, axis=1)
                y = data[TARGET_COL]
                if data_mode == 'agnostic':
                    if FILTER_MI:
                        X = filter_by_mutual_information(X, y, k=K_MI)
                    if FILTER_CORRELATION:
                        X = filter_by_correlation(X, threshold=CORR_THRESHOLD)
                    if FILTER_RF:
                        X = filter_by_rf_importance(X, y, k=K_RF)
                    if APPLY_PCA:
                        X = apply_pca(X, n_components=PCA_COMPONENTS)
                    y = y.loc[X.index]
                X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X.columns]
                # For exact reproduction, use per-split best parameters collected during training
                split_params_list = all_optimal_params.get('split_wise', {}).get(data_mode, {}).get(model_name, [])
                if not split_params_list:
                    # fallback to most common params
                    best_params_common = all_optimal_params.get(data_mode, {}).get(model_name, {})
                else:
                    best_params_common = None
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                aucs = []
                # Retrieve per-split params collected during training (if available)
                split_params_list = all_optimal_params.get('split_wise', {}).get(data_mode, {}).get(model_name, [])
                for split in range(N_SPLITS):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE + split, stratify=y)
                    # Determine params for this split
                    if split_params_list and split < len(split_params_list):
                        params_for_split = split_params_list[split] or {}
                    else:
                        # fallback to most-common params for this model/data_mode
                        params_for_split = all_optimal_params.get(data_mode, {}).get(model_name, {}) or {}

                    # Instantiate a fresh model with these params for reproducibility
                    try:
                        if model_name == 'RandomForest':
                            model_instance = RandomForestClassifier(random_state=RANDOM_STATE, **params_for_split)
                        elif model_name == 'GradientBoosting':
                            model_instance = GradientBoostingClassifier(random_state=RANDOM_STATE, **params_for_split)
                        elif model_name == 'LinearDiscriminantAnalysis':
                            model_instance = LinearDiscriminantAnalysis(**params_for_split)
                        elif model_name == 'LogisticRegression':
                            solver = params_for_split.get('solver', 'liblinear')
                            kwargs = {k: v for k, v in params_for_split.items() if k != 'solver'}
                            model_instance = LogisticRegression(random_state=42, solver=solver, **kwargs)
                        elif model_name == 'XGBoost':
                            model_instance = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', **params_for_split)
                        elif model_name == 'SVM':
                            model_instance = SVC(probability=True, random_state=RANDOM_STATE, **params_for_split)
                        else:
                            # unsupported model for ROC comparison
                            continue
                    except Exception as e:
                        # If instantiation with split params fails, fall back to default constructor
                        print(f"Warning: Failed to instantiate {model_name} with params {params_for_split} on split {split+1}: {e}")
                        if model_name == 'RandomForest':
                            model_instance = RandomForestClassifier(random_state=RANDOM_STATE)
                        elif model_name == 'GradientBoosting':
                            model_instance = GradientBoostingClassifier(random_state=RANDOM_STATE)
                        elif model_name == 'LinearDiscriminantAnalysis':
                            model_instance = LinearDiscriminantAnalysis()
                        elif model_name == 'LogisticRegression':
                            model_instance = LogisticRegression(random_state=42, solver='liblinear')
                        elif model_name == 'XGBoost':
                            model_instance = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
                        elif model_name == 'SVM':
                            model_instance = SVC(probability=True, random_state=RANDOM_STATE)

                    fitted_model = model_instance.fit(X_train, y_train)
                    y_proba = fitted_model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    aucs.append(roc_auc_score(y_test, y_proba))
                # If we have the exact fitted models for this data_mode/model_name, use them
                use_fitted_models = False
                if all_best_models.get(data_mode) and all_best_models[data_mode].get(model_name) and all_best_features.get(data_mode) and all_best_features[data_mode].get(model_name):
                    models_list = all_best_models[data_mode].get(model_name)
                    feats_list = all_best_features[data_mode].get(model_name)
                    if (isinstance(models_list, list) and isinstance(feats_list, list)
                        and len(models_list) >= N_SPLITS and len(feats_list) >= N_SPLITS
                        and all(m is not None for m in models_list[:N_SPLITS])
                        and all(f is not None for f in feats_list[:N_SPLITS])):
                        use_fitted_models = True

                if use_fitted_models:
                    tprs = []
                    aucs = []
                    for split in range(N_SPLITS):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE + split, stratify=y)
                        fitted_model = all_best_models[data_mode][model_name][split]
                        # Align test features to the feature set used when this model was trained
                        trained_cols = all_best_features.get(data_mode, {}).get(model_name, [None]*N_SPLITS)[split]
                        if trained_cols is None:
                            # Fallback: use X_test as-is
                            X_test_aligned = X_test
                        else:
                            # Reindex to trained columns; fill missing cols with 0
                            X_test_aligned = X_test.reindex(columns=trained_cols, fill_value=0)
                        y_proba = fitted_model.predict_proba(X_test_aligned)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        tprs.append(np.interp(mean_fpr, fpr, tpr))
                        tprs[-1][0] = 0.0
                        aucs.append(roc_auc_score(y_test, y_proba))
                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = np.mean(aucs)
                    std_auc = np.std(aucs)
                    plt.plot(mean_fpr, mean_tpr, lw=2, color=colors[data_mode], label=f'{data_type_labels[data_mode]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
                    std_tpr = np.std(tprs, axis=0)
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[data_mode], alpha=0.15)
                else:
                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = np.mean(aucs)
                    std_auc = np.std(aucs)
                    plt.plot(mean_fpr, mean_tpr, lw=2, color=colors[data_mode], label=f'{data_type_labels[data_mode]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
                    std_tpr = np.std(tprs, axis=0)
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[data_mode], alpha=0.15)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve Comparison for {model_name} ({N_SPLITS} Splits)', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'roc_compare_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()