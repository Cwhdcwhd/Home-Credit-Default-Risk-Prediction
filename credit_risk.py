"""
Home Credit Default Risk Prediction
This script builds a machine learning model to predict credit default risk
using various feature engineering techniques and LightGBM classifier.
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import pandas as pd
import numpy as np
import os
import sys
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, log_evaluation, early_stopping, plot_importance

# Set visualization style
plt.style.use('fivethirtyeight')

# ============================================================================
# SECTION 2: CONFIGURATION FLAGS
# ============================================================================

BaselineTraining = True
LogisticRegression = False
RandomForest = True
PolynomialFeaturesTesting = False
DomainFeaturesTesting = True

# Data directory path
default_dir = "C:\\桌面\\MATH 5470\\home-credit-default-risk\\"

# ============================================================================
# SECTION 3: UTILITY FUNCTIONS
# ============================================================================

def missing_values_table(df):
    """
    Generate a table showing missing values in the dataframe.
    
    Args:
        df (DataFrame): Input dataframe to analyze
        
    Returns:
        DataFrame: Table with missing values count and percentage
    """
    # Calculate missing values
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Create summary table
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    
    # Sort by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    
    # Print summary
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    
    return mis_val_table_ren_columns


def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage by optimizing data types.
    
    Args:
        df (DataFrame): Input dataframe
        verbose (bool): Print memory reduction statistics
        
    Returns:
        DataFrame: Optimized dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def plot_feature_importances(df):
    """
    Plot feature importances from a model.
    
    Args:
        df (DataFrame): Feature importances with 'feature' and 'importance' columns
        
    Returns:
        DataFrame: Sorted feature importances with normalized importance
    """
    # Sort features by importance
    df = df.sort_values('importance', ascending=False).reset_index()
    
    # Normalize importances to sum to 1
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Create horizontal bar chart (compact so it fits small windows)
    plt.figure(figsize=(4, 2))
    ax = plt.subplot()
    ax.barh(list(reversed(list(df.index[:10]))), 
            df['importance_normalized'].head(10), 
            align='center', edgecolor='k')
    
    # Set labels and formatting
    ax.set_yticks(list(reversed(list(df.index[:10]))))
    ax.set_yticklabels(df['feature'].head(10))
    
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances (Top 10)')
    plt.tight_layout()
    plt.show()
    
    return df


# ============================================================================
# SECTION 4: DATA LOADING FUNCTIONS
# ============================================================================

def get_balance_data():
    """
    Load and optimize balance-related datasets.
    
    Returns:
        tuple: (pos_bal, install, card_bal) DataFrames
    """
    # Define data types for memory optimization
    pos_dtype = {
        'SK_ID_PREV': np.uint32, 'SK_ID_CURR': np.uint32, 'MONTHS_BALANCE': np.int32,
        'SK_DPD': np.int32, 'SK_DPD_DEF': np.int32, 'CNT_INSTALMENT': np.float32,
        'CNT_INSTALMENT_FUTURE': np.float32
    }

    install_dtype = {
        'SK_ID_PREV': np.uint32, 'SK_ID_CURR': np.uint32, 'NUM_INSTALMENT_NUMBER': np.int32,
        'NUM_INSTALMENT_VERSION': np.float32, 'DAYS_INSTALMENT': np.float32,
        'DAYS_ENTRY_PAYMENT': np.float32, 'AMT_INSTALMENT': np.float32, 'AMT_PAYMENT': np.float32
    }

    card_dtype = {
        'SK_ID_PREV': np.uint32, 'SK_ID_CURR': np.uint32, 'MONTHS_BALANCE': np.int16,
        'AMT_CREDIT_LIMIT_ACTUAL': np.int32, 'CNT_DRAWINGS_CURRENT': np.int32,
        'SK_DPD': np.int32, 'SK_DPD_DEF': np.int32, 'AMT_BALANCE': np.float32,
        'AMT_DRAWINGS_ATM_CURRENT': np.float32, 'AMT_DRAWINGS_CURRENT': np.float32,
        'AMT_DRAWINGS_OTHER_CURRENT': np.float32, 'AMT_DRAWINGS_POS_CURRENT': np.float32,
        'AMT_INST_MIN_REGULARITY': np.float32, 'AMT_PAYMENT_CURRENT': np.float32,
        'AMT_PAYMENT_TOTAL_CURRENT': np.float32, 'AMT_RECEIVABLE_PRINCIPAL': np.float32,
        'AMT_RECIVABLE': np.float32, 'AMT_TOTAL_RECEIVABLE': np.float32,
        'CNT_DRAWINGS_ATM_CURRENT': np.float32, 'CNT_DRAWINGS_OTHER_CURRENT': np.float32,
        'CNT_DRAWINGS_POS_CURRENT': np.float32, 'CNT_INSTALMENT_MATURE_CUM': np.float32
    }

    # Load datasets
    pos_bal = pd.read_csv(os.path.join(default_dir, 'POS_CASH_balance.csv'), dtype=pos_dtype)
    install = pd.read_csv(os.path.join(default_dir, 'installments_payments.csv'), dtype=install_dtype)
    card_bal = pd.read_csv(os.path.join(default_dir, 'credit_card_balance.csv'), dtype=card_dtype)

    return pos_bal, install, card_bal


def get_dataset():
    """
    Load all datasets and optimize memory usage.
    
    Returns:
        tuple: (apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    """
    # Load application datasets
    app_train = pd.read_csv(os.path.join(default_dir, 'application_train.csv'))
    app_train = reduce_mem_usage(app_train)
    app_test = pd.read_csv(os.path.join(default_dir, 'application_test.csv'))
    app_test = reduce_mem_usage(app_test)
    apps = pd.concat([app_train, app_test])

    # Load supporting datasets
    prev = pd.read_csv(os.path.join(default_dir, 'previous_application.csv'))
    prev = reduce_mem_usage(prev)
    bureau = pd.read_csv(os.path.join(default_dir, 'bureau.csv'))
    bureau = reduce_mem_usage(bureau)
    bureau_bal = pd.read_csv(os.path.join(default_dir, 'bureau_balance.csv'))
    bureau_bal = reduce_mem_usage(bureau_bal)

    # Load balance data
    pos_bal, install, card_bal = get_balance_data()

    return apps, prev, bureau, bureau_bal, pos_bal, install, card_bal


# ============================================================================
# SECTION 5: FEATURE ENGINEERING - APPLICATION FEATURES
# ============================================================================

def get_apps_processed(apps):
    """
    Create engineered features from application data.
    
    Args:
        apps (DataFrame): Application data
        
    Returns:
        DataFrame: Applications with engineered features
    """
    # External source features
    apps['APPS_EXT_SOURCE_MEAN'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps['APPS_EXT_SOURCE_STD'].fillna(apps['APPS_EXT_SOURCE_STD'].mean())
    
    # Credit amount ratios
    apps['APPS_ANNUITY_CREDIT_RATIO'] = apps['AMT_ANNUITY'] / apps['AMT_CREDIT']
    apps['APPS_GOODS_CREDIT_RATIO'] = apps['AMT_GOODS_PRICE'] / apps['AMT_CREDIT']
    
    # Income ratios
    apps['APPS_ANNUITY_INCOME_RATIO'] = apps['AMT_ANNUITY'] / apps['AMT_INCOME_TOTAL']
    apps['APPS_CREDIT_INCOME_RATIO'] = apps['AMT_CREDIT'] / apps['AMT_INCOME_TOTAL']
    apps['APPS_GOODS_INCOME_RATIO'] = apps['AMT_GOODS_PRICE'] / apps['AMT_INCOME_TOTAL']
    apps['APPS_CNT_FAM_INCOME_RATIO'] = apps['AMT_INCOME_TOTAL'] / apps['CNT_FAM_MEMBERS']
    
    # Employment and birth day ratios
    apps['APPS_EMPLOYED_BIRTH_RATIO'] = apps['DAYS_EMPLOYED'] / apps['DAYS_BIRTH']
    apps['APPS_INCOME_EMPLOYED_RATIO'] = apps['AMT_INCOME_TOTAL'] / apps['DAYS_EMPLOYED']
    apps['APPS_INCOME_BIRTH_RATIO'] = apps['AMT_INCOME_TOTAL'] / apps['DAYS_BIRTH']
    apps['APPS_CAR_BIRTH_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_BIRTH']
    apps['APPS_CAR_EMPLOYED_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_EMPLOYED']
    
    return apps


# ============================================================================
# SECTION 6: FEATURE ENGINEERING - PREVIOUS APPLICATION FEATURES
# ============================================================================

def get_prev_processed(prev):
    """
    Create engineered features from previous application history.
    
    Args:
        prev (DataFrame): Previous applications
        
    Returns:
        DataFrame: Previous applications with engineered features
    """
    # Credit differences and ratios
    prev['PREV_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
    prev['PREV_CREDIT_APPL_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION']
    prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_APPLICATION']

    # Data cleansing - replace anomalous days
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Days difference
    prev['PREV_DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']

    # Calculate interest rate
    all_pay = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['PREV_INTERESTS_RATE'] = (all_pay / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

    return prev


def get_prev_amt_agg(prev):
    """
    Aggregate amount-related features from previous applications.
    
    Args:
        prev (DataFrame): Previous applications
        
    Returns:
        DataFrame: Aggregated features by SK_ID_CURR
    """
    agg_dict = {
        'SK_ID_CURR': ['count'],
        'AMT_CREDIT': ['mean', 'max', 'sum'],
        'AMT_ANNUITY': ['mean', 'max', 'sum'],
        'AMT_APPLICATION': ['mean', 'max', 'sum'],
        'AMT_DOWN_PAYMENT': ['mean', 'max', 'sum'],
        'AMT_GOODS_PRICE': ['mean', 'max', 'sum'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'PREV_CREDIT_DIFF': ['mean', 'max', 'sum'],
        'PREV_CREDIT_APPL_RATIO': ['mean', 'max'],
        'PREV_GOODS_DIFF': ['mean', 'max', 'sum'],
        'PREV_GOODS_APPL_RATIO': ['mean', 'max'],
        'PREV_DAYS_LAST_DUE_DIFF': ['mean', 'max', 'sum'],
        'PREV_INTERESTS_RATE': ['mean', 'max']
    }

    prev_group = prev.groupby('SK_ID_CURR')
    prev_amt_agg = prev_group.agg(agg_dict)
    prev_amt_agg.columns = ["PREV_" + "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]

    return prev_amt_agg


def get_prev_refused_appr_agg(prev):
    """
    Aggregate approval status from previous applications.
    
    Args:
        prev (DataFrame): Previous applications
        
    Returns:
        DataFrame: Count of approved and refused applications
    """
    # Filter for approved and refused applications
    prev_refused_appr_group = prev[prev['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])].groupby(
        ['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
    prev_refused_appr_agg = prev_refused_appr_group['SK_ID_CURR'].count().unstack()

    # Rename columns
    prev_refused_appr_agg.columns = ['PREV_APPROVED_COUNT', 'PREV_REFUSED_COUNT']
    prev_refused_appr_agg = prev_refused_appr_agg.fillna(0)

    return prev_refused_appr_agg


def get_prev_days365_agg(prev):
    """
    Aggregate features for previous applications from last year.
    
    Args:
        prev (DataFrame): Previous applications
        
    Returns:
        DataFrame: Aggregated features for recent applications
    """
    cond_days365 = prev['DAYS_DECISION'] > -365
    prev_days365_group = prev[cond_days365].groupby('SK_ID_CURR')
    agg_dict = {
        'SK_ID_CURR': ['count'],
        'AMT_CREDIT': ['mean', 'max', 'sum'],
        'AMT_ANNUITY': ['mean', 'max', 'sum'],
        'AMT_APPLICATION': ['mean', 'max', 'sum'],
        'AMT_DOWN_PAYMENT': ['mean', 'max', 'sum'],
        'AMT_GOODS_PRICE': ['mean', 'max', 'sum'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'PREV_CREDIT_DIFF': ['mean', 'max', 'sum'],
        'PREV_CREDIT_APPL_RATIO': ['mean', 'max'],
        'PREV_GOODS_DIFF': ['mean', 'max', 'sum'],
        'PREV_GOODS_APPL_RATIO': ['mean', 'max'],
        'PREV_DAYS_LAST_DUE_DIFF': ['mean', 'max', 'sum'],
        'PREV_INTERESTS_RATE': ['mean', 'max']
    }
    prev_days365_agg = prev_days365_group.agg(agg_dict)
    prev_days365_agg.columns = ["PREV_D365_" + "_".join(x).upper() for x in prev_days365_agg.columns.ravel()]

    return prev_days365_agg


def get_prev_agg(prev):
    """
    Combine all previous application aggregations.
    
    Args:
        prev (DataFrame): Previous applications
        
    Returns:
        DataFrame: Combined aggregated features
    """
    prev = get_prev_processed(prev)
    prev_amt_agg = get_prev_amt_agg(prev)
    prev_refused_appr_agg = get_prev_refused_appr_agg(prev)
    prev_days365_agg = get_prev_days365_agg(prev)
    
    # Merge all aggregations
    prev_agg = prev_amt_agg.merge(prev_refused_appr_agg, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_days365_agg, on='SK_ID_CURR', how='left')
    
    # Create approval ratios
    prev_agg['PREV_REFUSED_RATIO'] = prev_agg['PREV_REFUSED_COUNT'] / prev_agg['PREV_SK_ID_CURR_COUNT']
    prev_agg['PREV_APPROVED_RATIO'] = prev_agg['PREV_APPROVED_COUNT'] / prev_agg['PREV_SK_ID_CURR_COUNT']
    prev_agg = prev_agg.drop(['PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT'], axis=1)
    
    return prev_agg


# ============================================================================
# SECTION 7: FEATURE ENGINEERING - BUREAU FEATURES
# ============================================================================

def get_bureau_processed(bureau):
    """
    Create engineered features from bureau credit history.
    
    Args:
        bureau (DataFrame): Bureau data
        
    Returns:
        DataFrame: Bureau data with engineered features
    """
    # Day differences
    bureau['BUREAU_ENDDATE_FACT_DIFF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_FACT_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_ENDDATE_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
  
    # Debt ratios
    bureau['BUREAU_CREDIT_DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']
    bureau['BUREAU_CREDIT_DEBT_DIFF'] = bureau['AMT_CREDIT_SUM_DEBT'] - bureau['AMT_CREDIT_SUM']
    
    # Days past due indicators
    bureau['BUREAU_IS_DPD'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)
    bureau['BUREAU_IS_DPD_OVER120'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 120 else 0)
    
    return bureau


def get_bureau_day_amt_agg(bureau):
    """
    Aggregate all bureau features.
    
    Args:
        bureau (DataFrame): Bureau data
        
    Returns:
        DataFrame: Aggregated bureau features
    """
    bureau_agg_dict = {
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'BUREAU_ENDDATE_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
    }

    bureau_grp = bureau.groupby('SK_ID_CURR')
    bureau_day_amt_agg = bureau_grp.agg(bureau_agg_dict)
    bureau_day_amt_agg.columns = ['BUREAU_' + ('_').join(column).upper() for column in bureau_day_amt_agg.columns.ravel()]
    bureau_day_amt_agg = bureau_day_amt_agg.reset_index()
    
    return bureau_day_amt_agg


def get_bureau_active_agg(bureau):
    """
    Aggregate features for active bureau credits only.
    
    Args:
        bureau (DataFrame): Bureau data
        
    Returns:
        DataFrame: Aggregated active credit features
    """
    cond_active = bureau['CREDIT_ACTIVE'] == 'Active'
    bureau_active_grp = bureau[cond_active].groupby(['SK_ID_CURR'])
    bureau_agg_dict = {
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'BUREAU_ENDDATE_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
    }
    bureau_active_agg = bureau_active_grp.agg(bureau_agg_dict)
    bureau_active_agg.columns = ['BUREAU_ACT_' + ('_').join(column).upper() for column in bureau_active_agg.columns.ravel()]
    bureau_active_agg = bureau_active_agg.reset_index()
    
    return bureau_active_agg


def get_bureau_days750_agg(bureau):
    """
    Aggregate features for bureau credits from last 750 days.
    
    Args:
        bureau (DataFrame): Bureau data
        
    Returns:
        DataFrame: Aggregated features for recent credits
    """
    cond_days750 = bureau['DAYS_CREDIT'] > -750
    bureau_days750_group = bureau[cond_days750].groupby('SK_ID_CURR')
    bureau_agg_dict = {
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'BUREAU_ENDDATE_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
    }

    bureau_days750_agg = bureau_days750_group.agg(bureau_agg_dict)
    bureau_days750_agg.columns = ['BUREAU_ACT_' + ('_').join(column).upper() for column in bureau_days750_agg.columns.ravel()]
    bureau_days750_agg = bureau_days750_agg.reset_index()
    
    return bureau_days750_agg


def get_bureau_bal_agg(bureau, bureau_bal):
    """
    Aggregate features from bureau balance history.
    
    Args:
        bureau (DataFrame): Bureau data
        bureau_bal (DataFrame): Bureau balance data
        
    Returns:
        DataFrame: Aggregated bureau balance features
    """
    # Merge balance with bureau to get SK_ID_CURR
    bureau_bal = bureau_bal.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], on='SK_ID_BUREAU', how='left')
    
    # Create DPD indicators based on status
    bureau_bal['BUREAU_BAL_IS_DPD'] = bureau_bal['STATUS'].apply(lambda x: 1 if x in ['1', '2', '3', '4', '5'] else 0)
    bureau_bal['BUREAU_BAL_IS_DPD_OVER120'] = bureau_bal['STATUS'].apply(lambda x: 1 if x == '5' else 0)
    
    bureau_bal_grp = bureau_bal.groupby('SK_ID_CURR')
    bureau_bal_agg_dict = {
        'SK_ID_CURR': ['count'],
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'BUREAU_BAL_IS_DPD': ['mean', 'sum'],
        'BUREAU_BAL_IS_DPD_OVER120': ['mean', 'sum']
    }
    bureau_bal_agg = bureau_bal_grp.agg(bureau_bal_agg_dict)
    bureau_bal_agg.columns = ['BUREAU_BAL_' + ('_').join(column).upper() for column in bureau_bal_agg.columns.ravel()]
    bureau_bal_agg = bureau_bal_agg.reset_index()
    
    return bureau_bal_agg


def get_bureau_agg(bureau, bureau_bal):
    """
    Combine all bureau aggregations.
    
    Args:
        bureau (DataFrame): Bureau data
        bureau_bal (DataFrame): Bureau balance data
        
    Returns:
        DataFrame: Combined aggregated bureau features
    """
    bureau = get_bureau_processed(bureau)
    bureau_day_amt_agg = get_bureau_day_amt_agg(bureau)
    bureau_active_agg = get_bureau_active_agg(bureau)
    bureau_days750_agg = get_bureau_days750_agg(bureau)
    bureau_bal_agg = get_bureau_bal_agg(bureau, bureau_bal)
    
    # Merge all aggregations
    bureau_agg = bureau_day_amt_agg.merge(bureau_active_agg, on='SK_ID_CURR', how='left')
    bureau_agg['BUREAU_ACT_IS_DPD_RATIO'] = bureau_agg['BUREAU_ACT_BUREAU_IS_DPD_SUM'] / bureau_agg['BUREAU_SK_ID_BUREAU_COUNT']
    bureau_agg['BUREAU_ACT_IS_DPD_OVER120_RATIO'] = bureau_agg['BUREAU_ACT_BUREAU_IS_DPD_OVER120_SUM'] / bureau_agg['BUREAU_SK_ID_BUREAU_COUNT']
    
    bureau_agg = bureau_agg.merge(bureau_bal_agg, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_days750_agg, on='SK_ID_CURR', how='left')
    
    return bureau_agg


# ============================================================================
# SECTION 8: FEATURE ENGINEERING - BALANCE FEATURES
# ============================================================================

def get_pos_bal_agg(pos_bal):
    """
    Aggregate features from POS cash balance data.
    
    Args:
        pos_bal (DataFrame): POS balance data
        
    Returns:
        DataFrame: Aggregated POS balance features
    """
    # Create DPD indicators
    pos_bal['POS_IS_DPD'] = pos_bal['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    pos_bal['POS_IS_DPD_UNDER_120'] = pos_bal['SK_DPD'].apply(lambda x: 1 if (x > 0) and (x < 120) else 0)
    pos_bal['POS_IS_DPD_OVER_120'] = pos_bal['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)

    # Aggregate all POS data
    pos_bal_grp = pos_bal.groupby('SK_ID_CURR')
    pos_bal_agg_dict = {
        'SK_ID_CURR': ['count'],
        'MONTHS_BALANCE': ['min', 'mean', 'max'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
        'POS_IS_DPD': ['mean', 'sum'],
        'POS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'POS_IS_DPD_OVER_120': ['mean', 'sum']
    }
    pos_bal_agg = pos_bal_grp.agg(pos_bal_agg_dict)
    pos_bal_agg.columns = [('POS_') + ('_').join(column).upper() for column in pos_bal_agg.columns.ravel()]
    
    # Aggregate recent data (last 20 months)
    cond_months = pos_bal['MONTHS_BALANCE'] > -20
    pos_bal_m20_grp = pos_bal[cond_months].groupby('SK_ID_CURR')
    pos_bal_m20_agg_dict = {
        'SK_ID_CURR': ['count'],
        'MONTHS_BALANCE': ['min', 'mean', 'max'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
        'POS_IS_DPD': ['mean', 'sum'],
        'POS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'POS_IS_DPD_OVER_120': ['mean', 'sum']
    }

    pos_bal_m20_agg = pos_bal_m20_grp.agg(pos_bal_m20_agg_dict)
    pos_bal_m20_agg.columns = [('POS_M20') + ('_').join(column).upper() for column in pos_bal_m20_agg.columns.ravel()]
    pos_bal_agg = pos_bal_agg.merge(pos_bal_m20_agg, on='SK_ID_CURR', how='left')
    
    pos_bal_agg = pos_bal_agg.reset_index()
    
    return pos_bal_agg


def get_install_agg(install):
    """
    Aggregate features from installment payment data.
    
    Args:
        install (DataFrame): Installment data
        
    Returns:
        DataFrame: Aggregated installment features
    """
    # Create payment difference and ratio features
    install['AMT_DIFF'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
    install['AMT_RATIO'] = (install['AMT_PAYMENT'] + 1) / (install['AMT_INSTALMENT'] + 1)
    install['SK_DPD'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']

    # Create DPD indicators
    install['INS_IS_DPD'] = install['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    install['INS_IS_DPD_UNDER_120'] = install['SK_DPD'].apply(lambda x: 1 if (x > 0) and (x < 120) else 0)
    install['INS_IS_DPD_OVER_120'] = install['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)

    # Aggregate all installment data
    install_grp = install.groupby('SK_ID_CURR')
    install_agg_dict = {
        'SK_ID_CURR': ['count'],
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DAYS_ENTRY_PAYMENT': ['mean', 'max', 'sum'],
        'DAYS_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_PAYMENT': ['mean', 'max', 'sum'],
        'AMT_DIFF': ['mean', 'min', 'max', 'sum'],
        'AMT_RATIO': ['mean', 'max'],
        'SK_DPD': ['mean', 'min', 'max'],
        'INS_IS_DPD': ['mean', 'sum'],
        'INS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'INS_IS_DPD_OVER_120': ['mean', 'sum']
    }

    install_agg = install_grp.agg(install_agg_dict)
    install_agg.columns = ['INS_' + ('_').join(column).upper() for column in install_agg.columns.ravel()]

    # Aggregate recent data (last 365 days)
    cond_day = install['DAYS_ENTRY_PAYMENT'] >= -365
    install_d365_grp = install[cond_day].groupby('SK_ID_CURR')
    install_d365_agg_dict = {
        'SK_ID_CURR': ['count'],
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DAYS_ENTRY_PAYMENT': ['mean', 'max', 'sum'],
        'DAYS_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_PAYMENT': ['mean', 'max', 'sum'],
        'AMT_DIFF': ['mean', 'min', 'max', 'sum'],
        'AMT_RATIO': ['mean', 'max'],
        'SK_DPD': ['mean', 'min', 'max'],
        'INS_IS_DPD': ['mean', 'sum'],
        'INS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'INS_IS_DPD_OVER_120': ['mean', 'sum']
    }
    install_d365_agg = install_d365_grp.agg(install_d365_agg_dict)
    install_d365_agg.columns = ['INS_D365' + ('_').join(column).upper() for column in install_d365_agg.columns.ravel()]
    
    install_agg = install_agg.merge(install_d365_agg, on='SK_ID_CURR', how='left')
    install_agg = install_agg.reset_index()
    
    return install_agg


def get_card_bal_agg(card_bal):
    """
    Aggregate features from credit card balance data.
    
    Args:
        card_bal (DataFrame): Credit card balance data
        
    Returns:
        DataFrame: Aggregated credit card balance features
    """
    # Create ratio features
    card_bal['BALANCE_LIMIT_RATIO'] = card_bal['AMT_BALANCE'] / card_bal['AMT_CREDIT_LIMIT_ACTUAL']
    card_bal['DRAWING_LIMIT_RATIO'] = card_bal['AMT_DRAWINGS_CURRENT'] / card_bal['AMT_CREDIT_LIMIT_ACTUAL']

    # Create DPD indicators
    card_bal['CARD_IS_DPD'] = card_bal['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    card_bal['CARD_IS_DPD_UNDER_120'] = card_bal['SK_DPD'].apply(lambda x: 1 if (x > 0) and (x < 120) else 0)
    card_bal['CARD_IS_DPD_OVER_120'] = card_bal['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)

    # Aggregate all card balance data
    card_bal_grp = card_bal.groupby('SK_ID_CURR')
    card_bal_agg_dict = {
        'SK_ID_CURR': ['count'],
        'AMT_BALANCE': ['max'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'BALANCE_LIMIT_RATIO': ['min', 'max'],
        'DRAWING_LIMIT_RATIO': ['min', 'max'],
        'CARD_IS_DPD': ['mean', 'sum'],
        'CARD_IS_DPD_UNDER_120': ['mean', 'sum'],
        'CARD_IS_DPD_OVER_120': ['mean', 'sum']
    }
    card_bal_agg = card_bal_grp.agg(card_bal_agg_dict)
    card_bal_agg.columns = ['CARD_' + ('_').join(column).upper() for column in card_bal_agg.columns.ravel()]
    card_bal_agg = card_bal_agg.reset_index()
    
    # Aggregate recent data (last 3 months)
    cond_month = card_bal.MONTHS_BALANCE >= -3
    card_bal_m3_grp = card_bal[cond_month].groupby('SK_ID_CURR')
    card_bal_m3_agg = card_bal_m3_grp.agg(card_bal_agg_dict)
    card_bal_m3_agg.columns = ['CARD_M3' + ('_').join(column).upper() for column in card_bal_m3_agg.columns.ravel()]
    card_bal_agg = card_bal_agg.merge(card_bal_m3_agg, on='SK_ID_CURR', how='left')
    card_bal_agg = card_bal_agg.reset_index()
    
    return card_bal_agg


# ============================================================================
# SECTION 9: DATA PREPARATION AND INTEGRATION
# ============================================================================

def get_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal):
    """
    Combine all engineered features with application data.
    
    Args:
        apps, prev, bureau, bureau_bal, pos_bal, install, card_bal: All datasets
        
    Returns:
        DataFrame: Complete feature set with all aggregations
    """
    # Process and aggregate all data
    apps_all = get_apps_processed(apps)
    prev_agg = get_prev_agg(prev)
    bureau_agg = get_bureau_agg(bureau, bureau_bal)
    pos_bal_agg = get_pos_bal_agg(pos_bal)
    install_agg = get_install_agg(install)
    card_bal_agg = get_card_bal_agg(card_bal)
    
    print('prev_agg shape:', prev_agg.shape, 'bureau_agg shape:', bureau_agg.shape)
    print('pos_bal_agg shape:', pos_bal_agg.shape, 'install_agg shape:', install_agg.shape, 'card_bal_agg shape:', card_bal_agg.shape)
    print('apps_all before merge shape:', apps_all.shape)

    # Merge all aggregated features
    apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(bureau_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(pos_bal_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(install_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(card_bal_agg, on='SK_ID_CURR', how='left')

    print('apps_all after merge with all shape:', apps_all.shape)

    return apps_all


def get_apps_all_encoded(apps_all):
    """
    Encode categorical features to numerical values.
    
    Args:
        apps_all (DataFrame): Feature dataframe
        
    Returns:
        DataFrame: Dataframe with encoded categorical features
    """
    object_columns = apps_all.dtypes[apps_all.dtypes == 'object'].index.tolist()
    for column in object_columns:
        apps_all[column] = pd.factorize(apps_all[column])[0]
    
    return apps_all


def get_apps_all_train_test(apps_all):
    """
    Split data into training and testing sets.
    
    Args:
        apps_all (DataFrame): Complete feature dataframe
        
    Returns:
        tuple: (apps_all_train, apps_all_test) DataFrames
    """
    apps_all_train = apps_all[~apps_all['TARGET'].isnull()]
    apps_all_test = apps_all[apps_all['TARGET'].isnull()]
    apps_all_test = apps_all_test.drop('TARGET', axis=1)
    return apps_all_train, apps_all_test


# ============================================================================
# SECTION 10: MODEL TRAINING
# ============================================================================

def train_apps_all(apps_all_train):
    """
    Train LightGBM classifier on the complete feature set.
    
    Args:
        apps_all_train (DataFrame): Training data with target
        
    Returns:
        LGBMClassifier: Trained model
    """
    # Prepare features and target
    ftr_app = apps_all_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    target_app = apps_all_train['TARGET']

    # Split into training and validation sets
    train_x, valid_x, train_y, valid_y = train_test_split(
        ftr_app, target_app, test_size=0.2, random_state=2020
    )
    print('train shape:', train_x.shape, 'valid shape:', valid_x.shape)
    
    # Initialize and train LightGBM model
    clf = LGBMClassifier(
        nthread=4,
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=11,
        num_leaves=58,
        colsample_bytree=0.613,
        subsample=0.708,
        max_bin=407,
        reg_alpha=3.564,
        reg_lambda=4.930,
        min_child_weight=6,
        min_child_samples=165,
    )

    # Train with early stopping
    clf.fit(
        train_x, train_y,
        eval_set=[(train_x, train_y), (valid_x, valid_y)],
        eval_metric='auc',
        callbacks=[log_evaluation(100), early_stopping(200)]
    )
    
    return clf


# ============================================================================
# SECTION 11: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting model training...")
    
    # Load all datasets
    print("Loading datasets...")
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = get_dataset()
    
    # Create complete feature set
    print("Creating engineered features...")
    apps_all = get_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    
    # Clean up memory
    del apps, prev, bureau, bureau_bal, pos_bal, install, card_bal
    gc.collect()

    # Encode categorical variables
    print("Encoding categorical features...")
    apps_all = get_apps_all_encoded(apps_all)

    # Split into training and testing
    print("Splitting data...")
    apps_all_train, apps_all_test = get_apps_all_train_test(apps_all)

    # Train model
    print("Training model...")
    clf = train_apps_all(apps_all_train)

    # Make predictions
    print("Making predictions...")
    ftr_app_test = apps_all_test.drop(['SK_ID_CURR'], axis=1)
    predictions = clf.predict_proba(ftr_app_test)[:, 1]
    
    # Create submission file
    submission = apps_all_test[['SK_ID_CURR']].copy()
    submission['TARGET'] = predictions
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

    # Plot feature importances
    print("Plotting feature importances...")
    plot_importance(clf, figsize=(16, 32), max_num_features=10)
    plt.show()
    
    print("Model training complete!")
