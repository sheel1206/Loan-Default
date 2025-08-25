import pandas as pd
import numpy as np
from scipy.stats import zscore

initial_df = pd.read_csv('data/train.csv')

def load_data(input_file):

  # read in DF
  df = pd.read_csv(f'{input_file}')
  df = df.drop(columns=['Unnamed: 0'])

  return df

"""
Function to calculate financial ratios for DF
"""

def calculate_financial_ratios(df):

    # set-up feature set
    feature_set = pd.DataFrame()
    feature_set['fs_year'] = df['fs_year']
    feature_set['id'] = df['id']
    feature_set['stmt_date'] = df['stmt_date']
    feature_set['legal_struct'] = df['legal_struct']
    feature_set['ateco_sector'] = df['ateco_sector']

    # size
    feature_set['total_assets'] = df['asst_tot']
    feature_set['total_equity'] = df['eqty_tot']

    # debt coverage
    feature_set['solvency_debt_ratio'] = (df['debt_st'] + df['debt_lt']) / df['asst_tot']
    feature_set['debt_to_equity_ratio'] = (df['debt_st'] + df['debt_lt']) / df['eqty_tot']
    feature_set['interest_coverage_ratio'] = df['ebitda'] / df['exp_financing']
    feature_set['debt_service_coverage_ratio'] = df['ebitda'] / (df['debt_st'] + df['debt_lt'])

    # leverage
    feature_set['leverage_ratio'] = df['asst_tot'] / df['eqty_tot']
    feature_set['lt_debt_to_capitalization_ratio'] = df['debt_lt'] / (df['debt_lt'] + df['eqty_tot'])

    # profitability
    feature_set['profit_margin_ratio'] = df['profit'] / df['rev_operating']
    feature_set['return_on_assets'] = df['roa']
    feature_set['return_on_equity'] = df['roe']
    feature_set['organizational_profitability_ratio'] = df['ebitda'] / df['asst_tot']

    # liquidity
    feature_set['current_ratio'] = df['asst_current'] / (df['AP_st'] + df['debt_st'])
    feature_set['quick_ratio'] = (df['cash_and_equiv'] + df['AR']) / (df['AP_st'] + df['debt_st'])
    feature_set['cash_ratio'] = df['cash_and_equiv'] / df['asst_tot']

    # activity
    feature_set['receivables_turnover_ratio'] = df['rev_operating'] / df['AR']
    feature_set['asset_turnover_ratio'] = df['rev_operating'] / df['asst_tot']
    feature_set['inventory_turnover_ratio'] = df['COGS'] / (df['asst_current'] - df['AR'] - df['cash_and_equiv'])

    # growth
    if 'asset_growth' in df.columns:
      feature_set['asset_growth'] = df['asset_growth']
    if 'profit_growth' in df.columns:
      feature_set['profit_growth'] = df['profit_growth']
    if 'revenue_growth' in df.columns:
      feature_set['revenue_growth'] = df['revenue_growth']

    return feature_set

"""
Function to calculate growth over time given a current & prior value
"""

def calculate_one_growth(current, prior, first_year):

    # first year: set growth to 0
    if first_year:
        return 0

    # if prior is NaN: set to NaN
    elif pd.isna(prior):
        return float('nan')

    # if prior is 0: return Nan
    elif prior == 0:
        return float('nan')

    # growth calculation
    else:
        return (current - prior) / prior


"""
Function to obtain growth-over-time financial ratios for feature set
"""

def calculate_growth_features(df):

  # sort DF
  df_growth = df.copy()
  df_growth = df_growth.sort_values(['id', 'fs_year'])

  # obtain prior year's assets, profit, revenue
  df_growth['prior_year_assets'] = df_growth.groupby('id')['asst_tot'].shift(1)
  df_growth['prior_year_profit'] = df_growth.groupby('id')['profit'].shift(1)
  df_growth['prior_year_revenue'] = df_growth.groupby('id')['rev_operating'].shift(1)

  # identify whether current row is a first year of ID
  df_growth['first_year'] = (df_growth['id'] != df_growth['id'].shift(1)).astype(int)

  # calculate growth features
  df_growth['asset_growth'] = df_growth.apply(lambda x: calculate_one_growth(x['asst_tot'], x['prior_year_assets'], x['first_year']), axis=1)
  df_growth['profit_growth'] = df_growth.apply(lambda x: calculate_one_growth(x['profit'], x['prior_year_profit'], x['first_year']), axis=1)
  df_growth['revenue_growth'] = df_growth.apply(lambda x: calculate_one_growth(x['rev_operating'], x['prior_year_revenue'], x['first_year']), axis=1)

  # drop unnecessary columns
  df_growth.drop(columns=['prior_year_assets', 'prior_year_profit', 'prior_year_revenue'], inplace=True)

  return df_growth


"""
Function to replace missing or null values in the DF for each record with the mean of that column up to & including a record's year
"""

def process_financial_data(df, start_year=2007, end_year=2013, impute_method='mean'):

    # step 1: Prepare data and handle ROE and ROA calculations
    df = df.replace('NaT', np.nan).copy()

    # calculate ROE for missing values where 'eqty_tot' is not zero
    df.loc[df['roe'].isnull() & (df['eqty_tot'] != 0), 'roe'] = (df['profit'] / df['eqty_tot']) * 100
    # calculate ROA for missing values
    df.loc[df['roa'].isnull(), 'roa'] = (df['prof_operations'] / df['asst_tot']) * 100

    # step 2: Generate the feature set using the predefined function
    feature_set = calculate_financial_ratios(df)

    # step 3: Replace infinite values with NaN
    feature_set.replace([np.inf, -np.inf], np.nan, inplace=True)

    # step 4: Filter by years and sort by 'fs_year'
    feature_set = feature_set[(feature_set['fs_year'] >= start_year) & (feature_set['fs_year'] <= end_year)]
    feature_set = feature_set.sort_values('fs_year')

    # step 5: Impute missing values by year
    for year in range(start_year, end_year + 1):
        yearly_data = feature_set[feature_set['fs_year'] <= year]
        columns_to_impute = feature_set.columns.difference(['fs_year', 'id', 'stmt_date', 'legal_struct', 'ateco_sector'])

        # choose imputation method
        impute_values = yearly_data[columns_to_impute].mean() if impute_method == 'mean' else yearly_data[columns_to_impute].median()

        # fill missing values for each year
        for col in columns_to_impute:
            feature_set.loc[feature_set['fs_year'] == year, col] = feature_set.loc[feature_set['fs_year'] == year, col].fillna(impute_values[col])

    # remove sorting by year by resetting the index order
    return feature_set

"""
Function to assign labels to the DF - 1 for default, 0 for non-default
"""

def label_default(df, feature_set):

  # convert 'stmt_date' and 'def_date' to datetime format
  df['stmt_date'] = pd.to_datetime(df['stmt_date'], format='%Y-%m-%d')
  df['def_date'] = pd.to_datetime(df['def_date'], format='%d/%m/%Y', errors='coerce')

  # initialize label column with 0's
  df['label'] = 0

  for index, row in df.iterrows():
    if pd.notnull(row['def_date']):
      # calculate 6 months and 1 year 6 months after 'stmt_date'
      start_date = row['stmt_date'] + pd.DateOffset(months=6)
      end_date = row['stmt_date'] + pd.DateOffset(months=18)
      # check if 'def_date' falls within this range
      if start_date <= row['def_date'] <= end_date:
        df.at[index, 'label'] = 1

  # set labels in feature set
  feature_set['label'] = df['label']
  feature_set['fs_year'] = df['fs_year']

  return feature_set


"""
Function to obtain a list of numeric features in DF to apply transformations on
"""

def features_to_transform(df):

  # obtain features from DF
  features = list(df.columns)
  features = [feature for feature in features if feature not in ('fs_year', 'id', 'stmt_date', 'legal_struct', 'ateco_sector', 'label')]

  return features

"""
Function to bound outliers in DF
"""

def handle_outliers(df, columns):

    # clip outliers under 1% or greater than 99%
    for col in columns:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df

"""
Function to remove records that don't meet accounting principles (i.e. where assets < equity)
"""

def remove_errors(df):

    # remove records where assets < equity
    if 'leverage_ratio' in df.columns:
        cond = (df['leverage_ratio'] < 1) & (df['leverage_ratio'] > 0) & (df['fs_year'] != 2013)
        new_df = df[~cond]

    return new_df

"""
Function to apply log / sqrt transformations to given features in DF
"""

def apply_transformations(df, features):

  df = df.copy()

  # all features
  '''
  'total_assets', 'total_equity', 'solvency_debt_ratio', 'debt_to_equity_ratio', 'interest_coverage_ratio', 'debt_service_coverage_ratio', 'leverage_ratio',
  'lt_debt_to_capitalization_ratio', 'profit_margin_ratio', 'return_on_assets', 'return_on_equity', 'organizational_profitability_ratio', 'current_ratio',
  'quick_ratio', 'cash_ratio', 'receivables_turnover_ratio', 'asset_turnover_ratio', 'inventory_turnover_ratio'
  '''

  # transformations
  for feature in features:

    # handling values less than 0 in transformation
    min_value = df[feature].min()
    df[feature] = df[feature] - min_value + 1

    # now: applying transformations
    if feature == 'total_assets':
      df[feature] = np.log1p(df[feature])
    elif feature == 'total_equity':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'solvency_debt_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'debt_to_equity_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'interest_coverage_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'debt_service_coverage_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'leverage_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'lt_debt_to_capitalization_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'profit_margin_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'return_on_assets':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'return_on_equity':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'organizational_profitability_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'current_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'quick_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'cash_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'receivables_turnover_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'asset_turnover_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'inventory_turnover_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'asset_growth':
      df[feature] = np.log1p(df[feature])
    elif feature == 'profit_growth':
      df[feature] = np.log1p(df[feature])
    elif feature == 'revenue_growth':
      df[feature] = np.log1p(df[feature])
    else:
      print('feature not found')

  return df

"""
Function to apply standardization to given features in feature set
"""

def standardize_df(df, features):

  # standardize features
  for feature in features:
    df[feature] = zscore(df[feature])

  return df

"""
PRE-PROCESSING HARNESS: takes in a filepath to data, outputs a final feature set
"""

def preprocess_data(input_file):

  # step 1: load training data
  df = load_data(f'{input_file}')

  combined_df =  pd.concat([initial_df, df], ignore_index=True)
  print('step 1 complete')

  # step 2: calculate growth-over-time features
  df_growth = calculate_growth_features(combined_df)
  print('step 2 complete')
  
  # step 3: handle missing values
  feature_set = process_financial_data(df_growth)
  print('step 3 complete')
  
  # step 4: label data - SKIP

  # step 5: obtain quantitative features
  features = features_to_transform(feature_set)
  print('step 5 complete')

  # step 6: bound outliers
  final_df = handle_outliers(feature_set, features)
  print('step 6 complete')

  # step 7: remove errors
  final_df = remove_errors(final_df)
  print('step 7 complete')
  # d(feature_set)

  # step 8: apply transformations
  final_df = apply_transformations(final_df, features)
  print('step 8 complete')
 
  # step 9: standardize
  final_df = standardize_df(final_df, features)
  print('step 9 complete')

  # obtain holdout data
  final_df_holdout = final_df[final_df['fs_year'] == 2013]
 
  return final_df_holdout


"""
Function to output list of selected features for each model
"""

def obtain_features(model='logit'):

  # features found through univariate & multivariate analysis
  if model == 'logit':
    selected_features = ["total_equity", "debt_service_coverage_ratio", "leverage_ratio", "return_on_assets", "return_on_equity",  "receivables_turnover_ratio", "asset_growth", "profit_growth", "revenue_growth"]
    # selected_features = ['total_equity', 'solvency_debt_ratio', 'leverage_ratio', 'return_on_assets', 'receivables_turnover_ratio', 'profit_growth']
  elif model == 'gb':
    selected_features = ["total_equity", "debt_service_coverage_ratio", "leverage_ratio", "return_on_assets", "return_on_equity",  "receivables_turnover_ratio", "asset_growth", "profit_growth", "revenue_growth"]
    #selected_features = ['total_equity', 'solvency_debt_ratio', 'leverage_ratio', 'return_on_assets', 'receivables_turnover_ratio', 'profit_growth']
  elif model == 'rf':
    selected_features = []
  else:
    print('model not found')

  return selected_features


"""
Function to only use selected features for our dataset based on model
"""

def feature_selection(df, model='gb'):

  # obtain features depending on the model
  features = obtain_features(model)

  # crop feature set
  if model == 'logit':
    df_selected = df[features]
  elif model == 'gb':
    #selected_features = []
    df_selected = df[features]
  elif model == 'rf':
    selected_features = []
    df_selected = df[selected_features]
  else:
    print('model not found')

  return df_selected