import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin


def filter_data_date(data, reference_date):
    """For each person (record_id), get only the row with the visit date closest to reference_date."""
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    data = data.dropna(subset=['visit_date'], ignore_index=True)
    data = data[data['visit_date'] < reference_date]
    return data.sort_values(by=['record_id', 'visit_date'], ascending=[True, False]).drop_duplicates(subset=['record_id'], keep='first')
class DropAllMissingColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely missing in the training fold.

    This avoids SimpleImputer warnings/errors when a column has no observed values
    in a particular split (even if it exists in the full dataset).
    """

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.keep_columns_ = X.columns[~X.isna().all(axis=0)].tolist()
            self.keep_mask_ = None
        else:
            arr = np.asarray(X)
            # assumes missing values are represented as NaN
            self.keep_mask_ = ~np.all(np.isnan(arr), axis=0)
            self.keep_columns_ = None
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            keep = getattr(self, "keep_columns_", None)
            if keep is None:
                return X
            return X.loc[:, keep]
        arr = np.asarray(X)
        mask = getattr(self, "keep_mask_", None)
        if mask is None:
            return arr
        return arr[:, mask]

def _binary_encode(series: pd.Series) -> pd.Series:
    """Stable binary encoding without relying on `unique()` order.

    If >2 unique values exist, falls back to categorical codes.
    Missing values are preserved.
    """

    non_null = series.dropna()
    uniques = sorted(non_null.unique().tolist())
    if len(uniques) == 1:
        mapping = {uniques[0]: 0}
        return series.map(mapping)
    if len(uniques) == 2:
        mapping = {uniques[0]: 0, uniques[1]: 1}
        return series.map(mapping)
    return series.astype("category").cat.codes.replace({-1: np.nan})


def _days_before(reference_date, date_series, *, date_format: str | None = "%Y-%m-%d") -> pd.Series:
    """Convert a date-like series to (reference_date - date).days, preserving missing as NaN."""

    if date_format is None:
        dt = pd.to_datetime(date_series, errors="coerce")
    else:
        dt = pd.to_datetime(date_series, format=date_format, errors="coerce")
    return (reference_date - dt).dt.days

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



def apply_pca(X, n_components=None):
    """Applies Principal Component Analysis (PCA) to the feature matrix."""
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA applied. Explained variance ratio with {pca.n_components_} components: {pca.explained_variance_ratio_.sum():.4f}")
    pca_cols = [f'PC_{i+1}' for i in range(pca.n_components_)]
    return pd.DataFrame(X_pca, index=X.index, columns=pca_cols)


def create_features_background(df, reference_date):
    """Creates the comprehensive feature set from the 'background' analysis."""
    data = pd.DataFrame(index=df.index)
    #data['study'] = _binary_encode(df['study'])
    data['final_outcome_amp'] = df['final_outcome_amp']
    #data['index_date'] = (reference_date - pd.to_datetime(df['index_date'], format='%Y-%m-%d')).dt.days

    most_recent_vaccine = df[['vaccine_date_4', 'vaccine_date_3', 'vaccine_date_2', 'vaccine_date_1']].bfill(axis=1).iloc[:, 0]
    data['most_recent_vaccine'] = _days_before(reference_date, most_recent_vaccine)

    most_recent_infection = df[['pcr_pos_date_2', 'pcr_pos_date_1']].bfill(axis=1).iloc[:, 0]
    data['most_recent_infection'] = _days_before(reference_date, most_recent_infection)

    most_recent_contact = df[['pcr_pos_date_2', 'pcr_pos_date_1', 'vaccine_date_4', 'vaccine_date_3', 'vaccine_date_2', 'vaccine_date_1']].bfill(axis=1).iloc[:, 0]
    data['most_recent_contact'] = _days_before(reference_date, most_recent_contact)

    data['pcr_pos_date'] = _days_before(reference_date, df['pcr_pos_date_1'])
    data['age'] = pd.to_numeric(df['age'], errors="coerce")
    data['bmi'] = pd.to_numeric(df['bmi'], errors="coerce")
    pcr_dt = pd.to_datetime(df['pcr_pos_date_1'], format='%Y-%m-%d', errors="coerce")
    vac1_dt = pd.to_datetime(df['vaccine_date_1'], format='%Y-%m-%d', errors="coerce")
    data['delta_t'] = (pcr_dt - vac1_dt).dt.days
    data['comorbidity'] = _binary_encode(df['comorbidity'])
    data['first_exposure_date'] = _days_before(reference_date, df['first_exposure_date'])
    data['num_vaccines'] = df[['vaccine_date_1', 'vaccine_date_2', 'vaccine_date_3', 'vaccine_date_4']].notna().sum(axis=1)
    data['ab_chuv_igg_s_logratio'] = pd.to_numeric(df['ab_chuv_igg_s_logratio'], errors="coerce")
    data['ab_chuv_igg_n_logratio'] = pd.to_numeric(df['ab_chuv_igg_n_logratio'], errors="coerce")
    data['ab_chuv_iga_logratio'] = pd.to_numeric(df['ab_chuv_iga_logratio'], errors="coerce")
    data['first_exposure'] = _binary_encode(df['first_exposure'])
    data['prior_exposure'] = _binary_encode(df['prior_exposure'])
    data['last_antibody_before_omicron_igg_n_logratio'] = pd.to_numeric(df['last_antibody_before_omicron_igg_n_logratio'], errors="coerce")
    return data


def create_features_agnostic(df, reference_date):
    """
    Preprocess the COVID-19 data by converting categorical features to numerical,
    handling missing values, and creating additional features.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Processed data ready for modeling
    """
    data = pd.get_dummies(df, columns=['timepoint'], dtype=int)
    #data['study'] = _binary_encode(data['study'])
    data = pd.get_dummies(data, columns=['pop_sample'], dtype=int)
    
    # Process dates
    data['index_date'] = _days_before(reference_date, data['index_date'])
    
    # Process demographic information
    data['age'] = pd.to_numeric(data['age'], errors="coerce")
    data['sex'] = _binary_encode(data['sex'])
    data['bmi'] = pd.to_numeric(data['bmi'], errors="coerce")
    data = pd.get_dummies(data, columns=['smoking'], dtype=int)
    
    # Process medical conditions
    data['comorbidity'] = _binary_encode(data['comorbidity'])
    
    # Process various medical conditions with missing value handling
    for condition in ['hypertension', 'diabetes', 'cvd', 'respiratory', 'ckd', 'cancer', 'immune_supp']:
        data[condition] = _binary_encode(data[condition])
    
    # Process socioeconomic factors
    data = pd.get_dummies(data, columns=['income_3l'], dummy_na=0, dtype=int)
    data = pd.get_dummies(data, columns=['employment_4l'], dtype=int)
    data = pd.get_dummies(data, columns=['education_4l'], prefix='education', dummy_na=0, dtype=int)
    data['nationality'] = data['nationality'].map({'Non-Swiss': 0, 'Swiss': 1})
    data['summary_bl_behaviour'] = pd.to_numeric(data['summary_bl_behaviour'], errors="coerce")
    
    # Process symptoms
    data['symp_init'] = data['symp_init'].map({"No": 0, "Yes": 1})
    data = pd.get_dummies(data, columns=['symp_count_init_3l'], dummy_na=0, dtype=int)
    data['symp_sev_init_3l'] = data['symp_sev_init_3l'].map({"Mild to moderate": 0, "Severe to very severe": 1})
    
    # Process hospitalization
    data['hosp_2wks'] = data['hosp_2wks'].map({"No": 0, "Yes": 1})
    data['icu_2wks'] = data['icu_2wks'].map({"No": 0, "Yes": 1})
    
    # Process serological status
    data['seropos_at_bl'] = data['seropos_at_bl'].map({"No": 0, "Yes": 1})
    data['prior_pos_pcr'] = data['prior_pos_pcr'].map({"No": 0, "Yes": 1})
    data['prior_exposure'] = data['prior_exposure'].map({"No": 0, "Yes": 1})
    
    # Process exposure and vaccine data
    data['first_exposure_date'] = _days_before(reference_date, data['first_exposure_date'])
    data['first_exposure'] = _binary_encode(data['first_exposure'])
    
    # Process vaccine information
    data = pd.get_dummies(data, columns=['vaccine_type_1'], dummy_na=0, dtype=int)
    data = pd.get_dummies(data, columns=['vaccine_type_2'], dummy_na=0, dtype=int)
    data = pd.get_dummies(data, columns=['vaccine_type_3'], dummy_na=0, dtype=int)
    data = pd.get_dummies(data, columns=['vaccine_type_4'], dummy_na=0, dtype=int)
    
    # Process PCR test data
    data['pcr_pos_date_1'] = _days_before(reference_date, data['pcr_pos_date_1'])
    data['pcr_pos_date_2'] = _days_before(reference_date, data['pcr_pos_date_2'])
    data = pd.get_dummies(data, columns=['pcr_pos_sev_1'], dummy_na=0, dtype=int)
    
    # Process antibody data
    data['ab_chuv_iga_ratio'] = pd.to_numeric(data['ab_chuv_iga_ratio'], errors="coerce")
    data['ab_chuv_iga_result'] = _binary_encode(data['ab_chuv_iga_result'])
    data['ab_chuv_igg_s_ratio'] = pd.to_numeric(data['ab_chuv_igg_s_ratio'], errors="coerce")
    data['ab_chuv_igg_s_result'] = _binary_encode(data['ab_chuv_igg_s_result'])
    data['ab_chuv_igg_n_ratio'] = pd.to_numeric(data['ab_chuv_igg_n_ratio'], errors="coerce")
    data['ab_chuv_igg_n_result'] = _binary_encode(data['ab_chuv_igg_n_result'])
    data['ab_chuv_iga_logratio'] = pd.to_numeric(data['ab_chuv_iga_logratio'], errors="coerce")
    data['ab_chuv_igg_s_logratio'] = pd.to_numeric(data['ab_chuv_igg_s_logratio'], errors="coerce")
    data['ab_chuv_igg_n_logratio'] = pd.to_numeric(data['ab_chuv_igg_n_logratio'], errors="coerce")
    
    # Drop the ratio columns
    data.drop(columns=['ab_chuv_iga_ratio', 'ab_chuv_igg_s_ratio', 'ab_chuv_igg_n_ratio'], inplace=True)
    
    data['last_antibody_before_omicron_iga_logratio'] = pd.to_numeric(data['last_antibody_before_omicron_iga_logratio'], errors="coerce")
    data['last_antibody_before_omicron_igg_n_logratio'] = pd.to_numeric(data['last_antibody_before_omicron_igg_n_logratio'], errors="coerce")
    data['last_antibody_before_omicron_igg_s_logratio'] = pd.to_numeric(data['last_antibody_before_omicron_igg_s_logratio'], errors="coerce")
    
    # Process additional data
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', inplace=True, axis=1)
        
    for vac_num in range(1, 5):
        data[f'vaccine_date_{vac_num}'] = _days_before(reference_date, data[f'vaccine_date_{vac_num}'])
    
    # Fill missing behavior data
    for col in ['prior_hyg', 'prior_dist', 'prior_mask_mand']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    
    data['final_outcome_amp'] = df['final_outcome_amp']
    data['index_date'] = _days_before(reference_date, df['index_date'])
    return data

def create_features_sequence(df_main, df_sequence, reference_date):
    """Creates and combines features for the 'sequence' analysis."""
    data = create_features_background(df_main, reference_date)

    data_seq = pd.DataFrame(index=df_sequence.index)
    data_seq = pd.get_dummies(df_sequence, columns=['sequence_cat', 'sequence'], dtype=int)
    data_seq['days_between_lastexp'] = pd.to_numeric(df_sequence['days_between_lastexp'], errors="coerce")
    data_seq['days_betweentp'] = pd.to_numeric(df_sequence['days_betweentp'], errors="coerce")
    data_seq['behaviour_cat_2l_v2'] = _binary_encode(df_sequence['behaviour_cat_2l_v2'])
    
    return data.join(data_seq, how='left')

