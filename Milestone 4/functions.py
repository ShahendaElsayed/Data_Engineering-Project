import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
# from selenium import webdriver
# from selenium.webdriver.common.by import By
import time

from sqlalchemy import create_engine

engine = create_engine('postgresql://airflow:airflow@pgdatabase:5432/airflow')

# %%
# Replace values in the 'type' column
def columnsRename(fintech_indexed):
    fintech_indexed.columns = fintech_indexed.columns.str.lower().str.replace(' ', '_')
    fintech_indexed=fintech_indexed.set_index("customer_id")
    fintech_indexed['type'].replace({
        'INDIVIDUAL': 'Individual', 
        'JOINT': 'Joint App'
    }, inplace=True)
    return fintech_indexed

lookup_table = pd.DataFrame(columns=['Feature', 'Original Value', 'Imputed/Encoded Value'])

def track_changes(feature, original_value, imputed_value):
    global lookup_table
    new_row = pd.DataFrame({'Feature': [feature], 'Original Value': [original_value], 'Imputed/Encoded Value': [imputed_value]})
    lookup_table = pd.concat([lookup_table, new_row], ignore_index=True)


def handlingMissing(fintech_indexed):
    print('handling missing')
    #rate
    fintech_indexed.groupby('loan_status')['int_rate'].agg(
    missing_values=lambda x: x.isnull().sum(),
    mean_rate='mean'
    )
    fintech_indexed['rate_imputed'] = fintech_indexed.groupby('loan_status')['int_rate'].transform(lambda x: x.fillna(x.mean()))
    fintech_indexed['rate_imputed'].isnull().sum()
    mean_int_rate = fintech_indexed['rate_imputed'].mean()






    #annual inc joint
    fintech_indexed['annual_inc_joint'].isnull().sum()
    track_changes('annual_inc_joint', 'missing', "0")
    fintech_indexed['annual_inc_joint'] = fintech_indexed.groupby('loan_status')['annual_inc_joint'].transform(lambda x: x.fillna(fintech_indexed['annual_inc']))



    #description
    # Step 1: Standardize 'Description ' (e.g., convert to lowercase)
    fintech_indexed['description'] = fintech_indexed['description'].str.lower().str.replace(' ', '_')

    # Step 2: Impute missing 'description' with  'purpose'
    fintech_indexed['description'] = fintech_indexed['description'].fillna(fintech_indexed['purpose'])
    print(fintech_indexed['description'].isnull().sum())


    ##EMP TITLE
    income_bins = pd.qcut(fintech_indexed['annual_inc'], q=3, labels=['q1', 'q2', 'q3'])
    fintech_indexed['income_category'] = income_bins
    mode_low = fintech_indexed[fintech_indexed['income_category'] == 'q1']['emp_title'].mode()[0]
    mode_medium = fintech_indexed[fintech_indexed['income_category'] == 'q2']['emp_title'].mode()[0]
    mode_high = fintech_indexed[fintech_indexed['income_category'] == 'q3']['emp_title'].mode()[0]

    # Step 2: Create a mapping dictionary with modes for each category
    mode_mapping = {
        'q1': mode_low,
        'q2': mode_medium,
        'q3': mode_high
    }

    # Step 3: Use fillna() to impute missing 'emp_title' values based on the income category
    fintech_indexed['emp_title'] = fintech_indexed.apply(
        lambda row: row['emp_title'] if pd.notnull(row['emp_title']) else mode_mapping[row['income_category']],
        axis=1
    )
    empTitle_mode = fintech_indexed['emp_title'].mode()[0]



    ##EMP LENGTH
    # Step 1: Find the mode of 'emp_length'
    emp_length_mode = fintech_indexed['emp_length'].mode()[0]

    # Step 2: Impute missing 'emp_length' with the mode
    fintech_indexed['emp_length_imputed']=fintech_indexed['emp_length']
    fintech_indexed['emp_length_imputed'].fillna(emp_length_mode, inplace=True)
    track_changes('emp_length', 'missing', fintech_indexed['emp_length'].mode()[0]+ "(mode)")
    return fintech_indexed

def categorize_grade(numeric_value):
    if 1 <= numeric_value <= 5:
        return 'A'
    elif 6 <= numeric_value <= 10:
        return 'B'
    elif 11 <= numeric_value <= 15:
        return 'C'
    elif 16 <= numeric_value <= 20:
        return 'D'
    elif 21 <= numeric_value <= 25:
        return 'E'
    elif 26 <= numeric_value <= 30:
        return 'F'
    elif 31 <= numeric_value <= 35:
        return 'G'
    else:
        return 'Unknown'  # In case the value doesn't fall in any of these ranges

def categorize_grade(fintech_indexed):
    fintech_indexed['grade_category'] = fintech_indexed['grade'].apply(categorize_grade)
    return fintech_indexed

def outlier_annual_inc(fintech_out):
    print('handling outliers')
    log_annual = np.log(fintech_out['annual_inc'])
    fintech_out['annual_log']=fintech_out['annual_inc']
    Q1_annualInc = fintech_out['annual_log'].quantile(0.25)
    Q3_annualInc = fintech_out['annual_log'].quantile(0.75)
    IQR = Q3_annualInc - Q1_annualInc
    lower_bound_annual_inc = Q1_annualInc - 1.5 * IQR
    upper_bound_annual_inc = Q3_annualInc + 1.5 * IQR
    fintech_out['annual_log_capped'] = fintech_out['annual_log'].apply(
    lambda x: upper_bound_annual_inc if x > upper_bound_annual_inc else (lower_bound_annual_inc if x < lower_bound_annual_inc else x)
    )
    return fintech_out
def outlier_joint(fintech_out):
    #fintech_out['annual_inc_joint']=fintech_out['annual_inc_joint_imputed']
    log_annualj = np.log(fintech_out['annual_inc_joint'])
    fintech_out['annual_logj']=log_annualj
    Q1_annualInc = fintech_out['annual_logj'].quantile(0.25)
    Q3_annualInc = fintech_out['annual_logj'].quantile(0.75)
    IQR = Q3_annualInc - Q1_annualInc
    lower_bound_annual_incJ = Q1_annualInc - 1.5 * IQR
    upper_bound_annual_incJ = Q3_annualInc + 1.5 * IQR
    print(f"lower_bound_joint: {lower_bound_annual_incJ}")
    print(f"higher_bound_joint: {upper_bound_annual_incJ}")
    # Apply capping for the 'annual_logj' column
    fintech_out['annualjoint_log_capped'] = fintech_out['annual_logj'].apply(
        lambda x: upper_bound_annual_incJ if x > upper_bound_annual_incJ else (lower_bound_annual_incJ if x < lower_bound_annual_incJ else x)
    )
    fintech_out['annual_inc_joint_handled']=fintech_out['annualjoint_log_capped']
    # Display the result with capped values
    return fintech_out

def outlier_avg(fintech_out):
    log_avg_cur_bal = np.log(fintech_out['avg_cur_bal'] + 1)
    fintech_out['log_avg_cur_bal']=log_avg_cur_bal
    Q1_annualInc = fintech_out['log_avg_cur_bal'].quantile(0.25)
    Q3_annualInc = fintech_out['log_avg_cur_bal'].quantile(0.75)
    IQR = Q3_annualInc - Q1_annualInc
    lower_bound_annual_inc = Q1_annualInc - 1.5 * IQR
    upper_bound_annual_inc = Q3_annualInc + 1.5 * IQR
    q95 = fintech_out['log_avg_cur_bal'].quantile(0.95)
    q05 = fintech_out['log_avg_cur_bal'].quantile(0.05)
    fintech_out['avg_cur_bal_capped'] = fintech_out['log_avg_cur_bal'].apply(
    lambda x: q95 if x > q95 else (q05 if x < q05 else x)
    )
    fintech_out['avg_cur_bal_handled']=fintech_out['avg_cur_bal_capped']
    return fintech_out

def out_total(fintech_out):
    log_tot_cur_bal = np.log(fintech_out['tot_cur_bal'] + 1)  # Adding 1 to avoid log(0)
    fintech_out['log_tot_cur_bal']=log_tot_cur_bal
    Q1_annualInc = fintech_out['log_tot_cur_bal'].quantile(0.25)
    Q3_annualInc=fintech_out['log_tot_cur_bal'].quantile(0.75)
    IQR = Q3_annualInc - Q1_annualInc
    lower_bound_total = Q1_annualInc - 1.5 * IQR
    fintech_out['tot_cur_bal_capped'] = fintech_out['log_tot_cur_bal'].apply(
    lambda x: lower_bound_total if x < lower_bound_total else x
    )
    return fintech_out

def categorize_grade(numeric_value):
    if 1 <= numeric_value <= 5:
        return 'A'
    elif 6 <= numeric_value <= 10:
        return 'B'
    elif 11 <= numeric_value <= 15:
        return 'C'
    elif 16 <= numeric_value <= 20:
        return 'D'
    elif 21 <= numeric_value <= 25:
        return 'E'
    elif 26 <= numeric_value <= 30:
        return 'F'
    elif 31 <= numeric_value <= 35:
        return 'G'
    else:
        return 'Unknown' 

# %%
def transform1(fintech_indexed):
    print("transforming data")
    # Step 1: Convert the date column to datetime format
    fintech_indexed['issue_date'] = pd.to_datetime(fintech_indexed['issue_date'])  

    # Step 2: Extract the month number and create a new column
    fintech_indexed['month'] = fintech_indexed['issue_date'].dt.month
    fintech_indexed['salary_can_cover'] = fintech_indexed['annual_inc'] >= fintech_indexed['loan_amount']
    fintech_indexed['salary_can_cover'] = fintech_indexed['salary_can_cover'].astype(int)
    fintech_indexed['grade_category'] = fintech_indexed['grade'].apply(categorize_grade)
  
    #monthly installment
    fintech_indexed['loan_term'] = fintech_indexed['term'].str.extract('(\d+)').astype(int)
    P = fintech_indexed['loan_amount']  # Principal (loan amount)
    r = fintech_indexed['rate_imputed']/12  # Monthly interest rate = annual rate /12
    n = fintech_indexed['loan_term']  # Number of payments (months)

    fintech_indexed['monthly_installment'] = (P * (r * (1 + r) ** n)) / ((1 + r) ** n - 1)

    return fintech_indexed
    

# %%
def encode_home(fintech_indexed):
    print("encoding")
    fintech_indexed['home_ownership']=fintech_indexed['home_ownership'].str.lower()
    df_encoded_verification  = pd.get_dummies(fintech_indexed['home_ownership'], prefix='home_ownership')

    # Append the new columns to the existing dataframe
    fintech_indexed = pd.merge(fintech_indexed, df_encoded_verification, left_index=True, right_index=True)
    fintech_indexed[df_encoded_verification.columns] = fintech_indexed[df_encoded_verification.columns].astype(int)

    onehot_encoder = OneHotEncoder( drop='first')
    encoded_home_ownership = onehot_encoder.fit_transform(fintech_indexed[['home_ownership']])


    # Get the category names for One-Hot Encoded values to make the lookup table
    categories = onehot_encoder.categories_[0]
    classes_as_list = categories.tolist() 
    classes_metric = pd.DataFrame({
        'Metric': ["home_ownership_classes"],
        'Value': [repr(classes_as_list)]
    })
    for i, category in enumerate(categories[1:]):  # Skip the first category due to 'drop=first'
      track_changes('home_ownership', category, f'home_ownership_{category}')
    return fintech_indexed

# %%
# Define a function to extract numeric values from 'emp_length' strings
def convert_emp_length(emp_length):
    if '10+' in emp_length:
        return 11.0  # You can decide what value to give for '10+ years'
    elif '< 1' in emp_length:
        return 0  # For '< 1 year'
    else:
        match = re.search(r'\d+', emp_length)
        if match:
          return float(match.group()) 
        else : return None

# %%
def emp_length(fintech_indexed):
    fintech_indexed['emp_length']=fintech_indexed['emp_length_imputed']
    # Apply the function to the 'emp_length' column
    fintech_indexed['emp_length'] = fintech_indexed['emp_length'].apply(convert_emp_length)

    track_changes("emp_length","10+ years",11.0)
    track_changes("emp_length","<1 years",0.5)
    return fintech_indexed

# %%
def termAndPlan(fintech_indexed):
    fintech_indexed['term'] = fintech_indexed['term'].str.extract('(\d+)').astype(int)
    track_changes("term","36 month",36)
    track_changes("term","60 month",60)
    fintech_indexed['pymnt_plan']=fintech_indexed['pymnt_plan'].astype(int)
    track_changes("pymnt_plan","True",1)
    track_changes("pymnt_plan","False",0)
    return fintech_indexed

# %%
def grade_encode(fintech_indexed):
    label_encoder=LabelEncoder()
    fintech_indexed['grade_category_encoded'] = label_encoder.fit_transform(fintech_indexed['grade_category'])
    # Track each encoding dynamically for LabelEncoder
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    for original, encoded in mapping.items():
        track_changes('grade_category', original, encoded)
    return fintech_indexed
# %%
def encode_verification(fintech_indexed):  
    fintech_indexed['verification_status']=fintech_indexed['verification_status'].str.lower().str.replace(' ', '_')
    df_encoded_verification  = pd.get_dummies(fintech_indexed['verification_status'], prefix='verification')
    # Append the new columns to the existing dataframe
    fintech_indexed = pd.merge(fintech_indexed, df_encoded_verification, left_index=True, right_index=True)
    fintech_indexed[df_encoded_verification.columns] = fintech_indexed[df_encoded_verification.columns].astype(int)
    onehot_encoder = OneHotEncoder( drop='first')
    encoded_home_ownership = onehot_encoder.fit_transform(fintech_indexed[['verification_status']])
    # Get the category names for One-Hot Encoded values
    categories = onehot_encoder.categories_[0]
    classes_as_list = categories.tolist() 
    for i, category in enumerate(categories[1:]):  # Skip the first category due to 'drop=first'
        track_changes('verification_status', category, f'verification_status_{category}')
    return fintech_indexed

# %%
def encode_state(fintech_indexed):   
    label_encoder = LabelEncoder()
    fintech_indexed['state_encoded'] = label_encoder.fit_transform(fintech_indexed['state'])

    # Track each encoding dynamically for LabelEncoder
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    for original, encoded in mapping.items():
        track_changes('state', original, encoded)

    label_encoder = LabelEncoder()
    fintech_indexed['addr_state_encoded'] = label_encoder.fit_transform(fintech_indexed['addr_state'])
    classes_as_list = label_encoder.classes_.tolist()

    # Track each encoding dynamically for LabelEncoder
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    for original, encoded in mapping.items():
        track_changes('addr_state', original, encoded)
    return fintech_indexed
# %%


# %%
def encode_type(fintech_indexed):
    fintech_indexed['type']=fintech_indexed['type'].str.lower()
    df_encoded_type  = pd.get_dummies(fintech_indexed['type'], prefix='type')

    fintech_indexed = pd.merge(fintech_indexed, df_encoded_type, left_index=True, right_index=True)
    fintech_indexed[df_encoded_type.columns] = fintech_indexed[df_encoded_type.columns].astype(int)



    onehot_encoder = OneHotEncoder(drop='first')
    encoded_type= onehot_encoder.fit_transform(fintech_indexed[['type']])

    # Get the category names for One-Hot Encoded values
    categories = onehot_encoder.categories_[0]
    classes_as_list = categories.tolist() 
    
    for i, category in enumerate(categories[1:]):  # Skip the first category due to 'drop=first'
        track_changes('type', category, f'type_{category}')
    return fintech_indexed

# %%
def recompse_columns(fintech_indexed):
    fintech_indexed['annual_inc']=fintech_indexed['annual_log_capped']
    fintech_indexed['annual_inc_joint']=fintech_indexed['annual_inc_joint_handled']
    
    fintech_indexed['avg_cur_bal']=fintech_indexed['avg_cur_bal_handled']
    fintech_indexed['int_rate']=fintech_indexed['rate_imputed']
    fintech_indexed['tot_cur_bal']=fintech_indexed['tot_cur_bal_capped']
    return fintech_indexed

# %%

def normalization(fintech_indexed):
    scaler = MinMaxScaler(feature_range=(0, 15))

    # List of columns to normalize
    columns_to_normalize = ['loan_amount', 'funded_amount']

    # Apply Z-score normalization to the selected columns
    fintech_indexed[columns_to_normalize] = scaler.fit_transform(fintech_indexed[columns_to_normalize])
    return fintech_indexed

# %%
def drop(fintech_indexed):
    fintech_indexed=fintech_indexed.drop(columns=['income_category','emp_length_imputed','annual_log_capped','annual_inc_joint_handled','avg_cur_bal_handled','tot_cur_bal_capped','loan_term'])
    return fintech_indexed

def save(data_dir,fintech_indexed):
    print('saving')
    fintech_indexed.to_csv(data_dir + 'fintech_data_Met_P1_52_23665_clean.csv',index=False)
    lookup_table.to_csv(data_dir + 'lookup_table_Met_P1_52_23665.csv',index=False)

def extract_clean(file):
    df = pd.read_csv(file)
    df=columnsRename(df)
    df=handlingMissing(df)
    df=outlier_annual_inc(df)
    df=outlier_joint(df)
    df=outlier_avg(df)
    df=out_total(df)
    df.to_csv('/opt/airflow/data/fintech_clean.csv',index=False)
    print('loaded after cleaning succesfully')

def transform(file):
    df = pd.read_csv(file)
    df=transform1(df)
    df=encode_home(df)
    df=emp_length(df)
    df=termAndPlan(df)
    df=grade_encode(df)
    df=encode_verification(df)
    df=encode_state(df)
    df=encode_type(df)
    df=recompse_columns(df)
    ##df=normalization(df)
    df=drop(df)
    df.to_csv('/opt/airflow/data/fintech_transformed.csv',index=False)
    print('transformed succesfully')


def load_to_postgres(filename): 
    df = pd.read_csv(filename)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/fintech_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'fintech_clean',con = engine,if_exists='replace')
