#!/usr/bin/env python3
"""
COVID-19 Data Processing and Modeling Script

This script processes COVID-19 data and creates models to analyze trends and patterns.
It converts the functionality from models_new.ipynb notebook to a structured Python script.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, roc_curve, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Constants
DATA_DIR = 'data'
DATA_FILE = os.path.join(DATA_DIR, 'covid_data_2.csv')
REF_DATE = pd.to_datetime('2022-01-01')
target_col = 'final_outcome_amp'
SCALE = True
APPLY_PCA = False
FILTER_MI = True
RESULTS_DIR = 'results'

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)


def filter_data_date(data, reference_date=REF_DATE):
    """
    For each person (record_id), get only the row with the visit date closest to reference_date.
    
    Args:
        data (pd.DataFrame): Raw data containing visit dates
        reference_date (datetime): Reference date for filtering
        
    Returns:
        pd.DataFrame: Data filtered to include only the most relevant record per person
    """
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    data = data.dropna(subset=['visit_date'], ignore_index=True)
    data = data[data['visit_date'] < reference_date]
    return data.sort_values(by=['record_id', 'visit_date'], ascending=[True, False]).drop_duplicates(
        subset=['record_id'], keep='first')


def scale(data):
    """
    Scale numerical data (standardization).
    
    Args:
        data (pd.Series): Data to be scaled
        
    Returns:
        pd.Series: Scaled data
    """
    return data  #returns unscaled data
    # To implement real scaling, uncomment the following line:
    # return (data - data.mean()) / data.std()


def preprocess_data(df):
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


def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple machine learning models using nested cross-validation.
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target variable
        
    Returns:
        tuple: (best_models, results)
    """
    
    # Use stratified K-Fold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define models and hyperparameter grids
    models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'bootstrap': [True, False],
            'n_estimators': [20, 100, 200, 500],
            'max_depth': [None, 10, 20, 40],
            'min_samples_split': [5, 7, 10, 12, 20, 50],
            'min_samples_leaf': [2, 5, 8],
            'class_weight': ['balanced', None]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'param_grid': {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [3, 5],
            'reg_lambda': [0, 0.1, 1],
            'min_child_weight': [1, 3, 5, 10],  
            'scale_pos_weight': [1, (len(y)-sum(y))/sum(y)]
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
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'max_depth': [2, 5, 15, 20],
            'min_samples_leaf': [2, 25, 50],
        }
    },
    'LinearDiscriminantAnalysis': {
        'model': LinearDiscriminantAnalysis(),
        'param_grid': {
            'solver': ['svd', 'lsqr', 'eigen'],
            'shrinkage': [None, 'auto', 0.1, 0.5, 0.9],
            'tol': [1e-4, 1e-3, 1e-2],
        }
    }}
    
    # Outer loop for multiple train-test splits
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    best_models = {}
    accuracy_scores = {}
    balanced_accuracy_scores = {}
    feature_importances = {}
    
    # Store ROC curve data for plotting
    roc_data = {
        'mean_fpr': np.linspace(0, 1, 100)
    }
    
    for model_name, config in models.items():
        print(f"\nTraining and evaluating {model_name}...")
        tprs = []
        aucs = []
        accuracies = []
        balanced_accuracies = []
        
        for train_index, test_index in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Inner loop for GridSearchCV
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['param_grid'],
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Store the best model from the final fold
            if model_name not in best_models:
                best_models[model_name] = best_model
            
            # Get predictions and probabilities
            y_proba = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            tprs.append(np.interp(roc_data['mean_fpr'], fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc_score(y_test, y_proba))
            
            # Calculate accuracy metrics
            accuracies.append(accuracy_score(y_test, y_pred))
            balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
            
            # Save feature importances for supported models
            if hasattr(best_model, "feature_importances_"):
                if model_name not in feature_importances:
                    feature_importances[model_name] = []
                feature_importances[model_name].append(best_model.feature_importances_)
        
        # Calculate mean and std of tprs across folds
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        # Store ROC data for this model
        roc_data[model_name] = {
            'mean_tpr': mean_tpr,
            'std_tpr': std_tpr,
            'tprs_upper': tprs_upper,
            'tprs_lower': tprs_lower,
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs)
        }
        
        # Store accuracy metrics
        accuracy_scores[model_name] = (np.mean(accuracies), np.std(accuracies))
        balanced_accuracy_scores[model_name] = (np.mean(balanced_accuracies), np.std(balanced_accuracies))
        
        # Average feature importances if available
        if model_name in feature_importances:
            feature_importances[model_name] = np.mean(feature_importances[model_name], axis=0)
    
    return best_models, {
        'roc_data': roc_data,
        'accuracy_scores': accuracy_scores,
        'balanced_accuracy_scores': balanced_accuracy_scores,
        'feature_importances': feature_importances
    }


def visualize_results_with_cv(X, results, save=True):
    """
    Create and save visualizations of model results with cross-validation.
    
    Args:
        X (pd.DataFrame): Feature dataframe for column names
        results (dict): Results from model training and evaluation
        save (bool): Whether to save the plots
    """
    
    # Unpack results
    roc_data = results['roc_data']
    accuracy_scores = results['accuracy_scores']
    balanced_accuracy_scores = results['balanced_accuracy_scores']
    feature_importances = results['feature_importances']
    
    # 1. Plot ROC curves with confidence intervals
    plt.figure(figsize=(12, 10))
    for model_name in roc_data.keys():
        if model_name == 'mean_fpr':
            continue
            
        model_roc = roc_data[model_name]
        plt.plot(
            roc_data['mean_fpr'], 
            model_roc['mean_tpr'], 
            lw=2,
            label=f"{model_name} (AUC = {model_roc['mean_auc']:.2f} ± {model_roc['std_auc']:.2f})"
        )
        plt.fill_between(
            roc_data['mean_fpr'], 
            model_roc['tprs_lower'], 
            model_roc['tprs_upper'], 
            alpha=0.2
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves with Cross-Validation', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save:
        plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves_cv.png'), dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # 2. Plot feature importances for applicable models
    for model_name, importances in feature_importances.items():
        plt.figure(figsize=(14, 10))
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        features = X.columns
        
        # Plot top 20 features
        top_n = min(20, len(features))
        plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14)
        plt.bar(range(top_n), importances[indices][:top_n], align='center')
        plt.xticks(range(top_n), [features[i] for i in indices][:top_n], rotation=90)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(RESULTS_DIR, f'feature_importance_{model_name}.png'), 
                      dpi=300, bbox_inches='tight')
        
        plt.close()
    
    # 3. Plot accuracy comparison
    plt.figure(figsize=(12, 8))
    models = list(accuracy_scores.keys())
    x = np.arange(len(models))
    width = 0.35
    
    # Get means and errors
    acc_means = [accuracy_scores[m][0] for m in models]
    acc_stds = [accuracy_scores[m][1] for m in models]
    bal_acc_means = [balanced_accuracy_scores[m][0] for m in models]
    bal_acc_stds = [balanced_accuracy_scores[m][1] for m in models]
    
    # Create grouped bar chart
    plt.bar(x - width/2, acc_means, width, label='Accuracy', yerr=acc_stds, capsize=5)
    plt.bar(x + width/2, bal_acc_means, width, label='Balanced Accuracy', yerr=bal_acc_stds, capsize=5)
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.xticks(x, models, rotation=45)
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(RESULTS_DIR, 'model_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    
    plt.close()


def print_detailed_metrics(models, results):
    """
    Print detailed metrics for all models.
    
    Args:
        models (dict): Dictionary of trained models
        results (dict): Results from model training and evaluation
    """
    # Unpack results
    accuracy_scores = results['accuracy_scores']
    balanced_accuracy_scores = results['balanced_accuracy_scores']
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Accuracy':<15} {'Balanced Acc':<15} {'AUC':<15}")
    print("-"*65)
    
    for model_name in models.keys():
        acc_mean, acc_std = accuracy_scores[model_name]
        bal_acc_mean, bal_acc_std = balanced_accuracy_scores[model_name]
        auc_mean = results['roc_data'][model_name]['mean_auc']
        auc_std = results['roc_data'][model_name]['std_auc']
        
        print(f"{model_name:<20} "
              f"{acc_mean:.3f} ± {acc_std:.3f}  "
              f"{bal_acc_mean:.3f} ± {bal_acc_std:.3f}  "
              f"{auc_mean:.3f} ± {auc_std:.3f}")
    
    print("\n" + "="*80)
    print("BEST MODEL PARAMETERS")
    print("="*80)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        for param, value in model.get_params().items():
            if param in model.get_params() and not param.startswith('_'):
                print(f"  {param}: {value}")


def split_train_test_data(data, test_ratio=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        data (pd.DataFrame): Processed data
        test_ratio (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    # Extract target variable
    y = data[target_col]
    X = data.drop(columns=[target_col])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to orchestrate the COVID-19 data processing and modeling workflow.
    """
    print("Starting COVID-19 data analysis with advanced machine learning...")
    
    # Load data
    print("Loading data...")
    all_data = pd.read_csv(DATA_FILE)
    raw_data = filter_data_date(all_data).dropna(axis=1, how='all')  # Apply date filter and remove empty columns
    raw_data.pop('visit_date')
    raw_data.set_index('record_id', inplace=True)
    raw_data.columns = raw_data.columns.astype(str).str.replace(r'[\[\]<>]', '', regex=True)
    
    print(f"Loaded data shape after filtering: {raw_data.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_data(raw_data)
    print(processed_data.head())
    assert False
    print(f"Processed data shape: {processed_data.shape}")
    
    # Split into features and target
    print("Preparing features and target...")
    y = processed_data[target_col]
    X = processed_data.drop(columns=[target_col])
    
    # Train and evaluate models using nested cross-validation
    print("Training and evaluating models with nested cross-validation...")
    best_models, results = train_and_evaluate_models(X, y)
    
    # Print detailed metrics
    print_detailed_metrics(best_models, results)
    
    # Create and save visualizations
    print("Creating visualizations...")
    visualize_results_with_cv(X, results)
    
    print("Analysis complete! Results saved to the 'results' directory.")


if __name__ == "__main__":
    main()