import pdfplumber
import pandas as pd
import glob
import os
import numpy as np

folder_path = "C:\\Users\\buitu\\Coding Projects\\PD"

# Get all .csv files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Read and concatenate all CSVs
df_list = []
for file in csv_files:
    df = pd.read_csv(
        file,
        delimiter='|',
        header=None,
        dtype={105: 'object'},  # Treat column 105 as string
        low_memory=False
    )
    df_list.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True)

# print(f"Loaded {len(csv_files)} files with total shape: {combined_df.shape}")

field_names = []
file_path = "C:\\Users\\buitu\\Coding Projects\\PD\\crt-file-layout-and-glossary.pdf"
all_field_names = []
field_names_with_nan_sf = []

with pdfplumber.open(file_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if table:
            headers = table[0]
            # Clean headers (remove newlines and trim spaces)
            headers_clean = [h.replace('\n', ' ').strip() if h else '' for h in headers]
            
            try:
                field_name_idx = headers_clean.index("Field Name")
            except ValueError:
                # Skip if no "Field Name" column
                continue
            
            # Attempt to get SF Loan Performance column index, if it exists
            try:
                sf_loan_perf_idx = headers_clean.index("Single-Family (SF) Loan Performance")
            except ValueError:
                sf_loan_perf_idx = None  # No such column on this page

            for row in table[1:]:
                if len(row) > field_name_idx:
                    field_name = row[field_name_idx]
                    all_field_names.append(field_name)

                    if sf_loan_perf_idx is not None and len(row) > sf_loan_perf_idx:
                        sf_value = row[sf_loan_perf_idx]
                        sf_value_norm = sf_value.strip() if sf_value else ''
                        if sf_value_norm == '' or sf_value_norm.lower() == 'na':
                            field_names_with_nan_sf.append(field_name)

for i in range(len(all_field_names)):
    if "\n" in all_field_names[i]:
        all_field_names[i] = all_field_names[i].replace('\n', ' ')

combined_df.columns = all_field_names
column_names_lst = ["POOL_ID", "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "SERVICER",
                        "MASTER_SERVICER", "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "ISSUANCE_UPB",
                        "CURRENT_UPB", "ORIG_TERM", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE",
                        "REM_MONTHS", "ADJ_REM_MONTHS", "MATR_DT", "OLTV", "OCLTV",
                        "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE",
                        "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP", "MI_PCT",
                        "PRODUCT", "PPMT_FLG", "IO", "FIRST_PAY_IO", "MNTHS_TO_AMTZ_IO",
                        "DLQ_STATUS", "PMT_HISTORY", "MOD_FLAG", "MI_CANCEL_FLAG", "Zero_Bal_Code",
                        "ZB_DTE", "LAST_UPB", "RPRCH_DTE", "CURR_SCHD_PRNCPL", "TOT_SCHD_PRNCPL",
                        "UNSCHD_PRNCPL_CURR", "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
                        "DISPOSITION_DATE", "FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
                        "ASSET_RECOVERY_COSTS", "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
                        "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
                        "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
                        "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB", "PRINCIPAL_FORGIVENESS_AMOUNT",
                        "ORIGINAL_LIST_START_DATE", "ORIGINAL_LIST_PRICE", "CURRENT_LIST_START_DATE",
                        "CURRENT_LIST_PRICE", "ISSUE_SCOREB", "ISSUE_SCOREC", "CURR_SCOREB",
                        "CURR_SCOREC", "MI_TYPE", "SERV_IND", "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
                        "CUMULATIVE_MODIFICATION_LOSS_AMOUNT", "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                        "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS", "HOMEREADY_PROGRAM_INDICATOR",
                        "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", "RELOCATION_MORTGAGE_INDICATOR",
                        "ZERO_BALANCE_CODE_CHANGE_DATE", "LOAN_HOLDBACK_INDICATOR", "LOAN_HOLDBACK_EFFECTIVE_DATE",
                        "DELINQUENT_ACCRUED_INTEREST", "PROPERTY_INSPECTION_WAIVER_INDICATOR",
                        "HIGH_BALANCE_LOAN_INDICATOR", "ARM_5_YR_INDICATOR", "ARM_PRODUCT_TYPE",
                        "MONTHS_UNTIL_FIRST_PAYMENT_RESET", "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET",
                        "INTEREST_RATE_CHANGE_DATE", "PAYMENT_CHANGE_DATE", "ARM_INDEX",
                        "ARM_CAP_STRUCTURE", "INITIAL_INTEREST_RATE_CAP", "PERIODIC_INTEREST_RATE_CAP",
                        "LIFETIME_INTEREST_RATE_CAP", "MARGIN", "BALLOON_INDICATOR",
                        "PLAN_NUMBER", "FORBEARANCE_INDICATOR", "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
                        "DEAL_NAME", "RE_PROCS_FLAG", "ADR_TYPE", "ADR_COUNT", "ADR_UPB", "PAYMENT_DEFERRAL_MOD_EVENT_FLAG", "INTEREST_BEARING_UPB"]

name_dct = dict(zip(combined_df.columns, column_names_lst))
# R-style column types
lppub_column_classes = [
    "character", "character", "character", "character", "character", "character",
    "character", "numeric", "numeric", "numeric", "numeric",
    "numeric", "numeric", "character", "character", "numeric", "numeric",
    "numeric", "character", "numeric", "numeric", "character", "numeric",
    "numeric", "numeric", "character", "character", "character",
    "numeric", "character", "character", "character", "character",
    "numeric", "character", "character", "character", "character",
    "numeric", "character", "character", "character", "character",
    "character", "character", "numeric", "character", "numeric",
    "numeric", "numeric", "character", "character", "character",
    "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
    "numeric", "numeric", "numeric", "numeric", "numeric", "character",
    "numeric", "character", "numeric", "numeric", "numeric", "numeric",
    "numeric", "numeric", "character", "numeric", "numeric", "numeric",
    "numeric", "character", "numeric", "character", "numeric", "character",
    "numeric", "numeric", "character", "character", "numeric", "numeric",
    "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
    "numeric", "numeric", "numeric", "numeric", "numeric", "character",
    "character", "character", "character", "character",
    "character", "numeric", "numeric", "character", "numeric"
]

# Map R types to Python dtypes
r_to_python_type_map = {"character": "string", "numeric": "float64"}

# Create a dictionary: column_name -> dtype
column_dtypes = {
    col: r_to_python_type_map[rtype]
    for col, rtype in zip(combined_df.columns, lppub_column_classes)
}

# Assign data types
combined_df = combined_df.astype(column_dtypes)
combined_df.drop(columns=field_names_with_nan_sf, inplace=True, errors='ignore')

# Step 1: Identify columns with exactly 2 unique values including NaN, and one of them is NaN
cols_to_fill = combined_df.apply(
    lambda col: col.nunique(dropna=False) == 2 and col.isna().any()
)
cols_to_fill = cols_to_fill[cols_to_fill].index.tolist()

# Step 2: Fill NaN with the non-NaN unique value in each of these columns
for col in cols_to_fill:
    # Get the non-NaN unique value
    non_nan_values = combined_df[col].dropna().unique()
    
    if len(non_nan_values) == 1:
        fill_value = non_nan_values[0]
        combined_df[col] = combined_df[col].fillna(fill_value)

combined_df.drop(columns=["Payment Deferral Modification Event Indicator", "Remaining Months to Maturity"], inplace=True, errors='ignore')
cols_with_all_nan = combined_df.columns[
    combined_df.isna().all()]
combined_df.drop(columns=cols_with_all_nan.tolist(), inplace=True, errors='ignore')

cols_with_nan_or_empty = combined_df.columns[
    combined_df.isna().any() | (combined_df == '').any()
]

combined_df['Servicer Name'] = combined_df['Servicer Name'].fillna(combined_df['Servicer Name'].mode()[0])
combined_df['Current Interest Rate'] = combined_df['Current Interest Rate'].fillna(combined_df['Current Interest Rate'].mean())
combined_df['Loan Age'] = combined_df['Loan Age'].fillna(combined_df['Loan Age'].median())
combined_df['Remaining Months to Legal Maturity'] = combined_df['Remaining Months to Legal Maturity'].fillna(combined_df['Remaining Months to Legal Maturity'].mode()[0])
combined_df['Maturity Date'] = combined_df['Maturity Date'].fillna(combined_df['Maturity Date'].mode()[0])
combined_df['Debt-To-Income (DTI)'] = combined_df['Debt-To-Income (DTI)'].fillna(combined_df['Debt-To-Income (DTI)'].mean())
combined_df['Borrower Credit Score at Origination'] = combined_df['Borrower Credit Score at Origination'].fillna(combined_df['Borrower Credit Score at Origination'].mean())
combined_df['Co-Borrower Credit Score at Origination'] = combined_df['Co-Borrower Credit Score at Origination'].fillna(combined_df['Co-Borrower Credit Score at Origination'].mean())
combined_df['Zip Code Short'] = combined_df['Zip Code Short'].fillna(combined_df['Zip Code Short'].mode()[0])
combined_df['Mortgage Insurance Percentage'] = combined_df['Mortgage Insurance Percentage'].fillna(combined_df['Mortgage Insurance Percentage'].mean())
combined_df['Loan Payment History'] = combined_df['Loan Payment History'].fillna(combined_df['Loan Payment History'].mode()[0])
combined_df['Modification Flag'] = combined_df['Modification Flag'].fillna(combined_df['Modification Flag'].mode()[0])
combined_df['Zero Balance Effective Date'] = combined_df['Zero Balance Effective Date'].fillna(combined_df['Zero Balance Effective Date'].mode()[0])
combined_df['UPB at the Time of Removal'] = combined_df['UPB at the Time of Removal'].fillna(combined_df['UPB at the Time of Removal'].mean())
combined_df['Total Principal Current'] = combined_df['Total Principal Current'].fillna(combined_df['Total Principal Current'].mean())
combined_df['Last Paid Installment Date'] = combined_df['Last Paid Installment Date'].fillna(combined_df['Last Paid Installment Date'].mode()[0])
combined_df['Foreclosure Date'] = combined_df['Foreclosure Date'].fillna(combined_df['Foreclosure Date'].mode()[0])
combined_df['Modification-Related Non-Interest Bearing UPB'] = combined_df['Modification-Related Non-Interest Bearing UPB'].fillna(combined_df['Modification-Related Non-Interest Bearing UPB'].mean())
combined_df['Mortgage Insurance Type'] = combined_df['Mortgage Insurance Type'].fillna(combined_df['Mortgage Insurance Type'].mode()[0])
combined_df['Servicing Activity Indicator'] = combined_df['Servicing Activity Indicator'].fillna(combined_df['Servicing Activity Indicator'].mode()[0])
combined_df['Repurchase Make Whole Proceed'] = combined_df['Repurchase Make Whole Proceed'].fillna(combined_df['Repurchase Make Whole Proceed'].mode()[0])
combined_df['Total Deferral Amount'] = combined_df['Total Deferral Amount'].fillna(combined_df['Total Deferral Amount'].mean())

combined_df.rename(columns=name_dct, inplace=True)

quarter_to_month = {'Q1': '03', 'Q2': '06', 'Q3': '09', 'Q4': '12'}

start_idx = 0
for file, df in zip(csv_files, df_list):
    file_base = os.path.basename(file).replace('.csv', '')
    year = file_base[:4]
    qtr = file_base[4:]
    month = quarter_to_month.get(qtr, '12')
    acquisition_date = f"{year}-{month}-01"
    
    end_idx = start_idx + len(df)
    
    # Assign acquisition_date to the slice of combined_df corresponding to this file's rows
    combined_df.loc[start_idx:end_idx - 1, 'acquisition_date'] = acquisition_date
    
    start_idx = end_idx

# Convert 'MMYYYY' -> 'YYYY-MM-01' with explicit format
def mmYYYY_to_date(col):
    return pd.to_datetime(
        combined_df[col].str[2:6] + '-' + combined_df[col].str[0:2] + '-01',
        format='%Y-%m-%d',
        errors='coerce'  # Use this to safely handle bad formats
    )

for i in ['ACT_PERIOD', 'FIRST_PAY', 'ORIG_DATE']:
    combined_df[i] = combined_df[i].astype(str).str.zfill(6)
    combined_df[i] = mmYYYY_to_date(i)

# Step 3: Sort the data
combined_df = combined_df.sort_values(by=['LOAN_ID', 'ACT_PERIOD']).reset_index(drop=True)

# Rename relevant columns to match standardized names
acquisitionFile = combined_df[[
    'LOAN_ID', 'ACT_PERIOD', 'CHANNEL', 'SELLER', 'ORIG_RATE', 'ORIG_UPB',
    'ORIG_TERM', 'ORIG_DATE', 'FIRST_PAY', 'OLTV',
    'OCLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
    'FIRST_FLAG', 'PURPOSE', 'PROP', 'NO_UNITS', 'OCC_STAT',
    'STATE', 'ZIP', 'MI_PCT', 'PRODUCT', 'MI_TYPE',
    'RELOCATION_MORTGAGE_INDICATOR', 'acquisition_date'
]].rename(columns={
    'CHANNEL': 'ORIG_CHN',
    'ORIG_RATE': 'orig_rt',
    'ORIG_UPB': 'orig_amt',
    'ORIG_TERM': 'orig_trm',
    'ORIG_DATE': 'orig_date',
    'FIRST_PAY': 'first_pay',
    'OLTV': 'oltv',
    'OCLTV': 'ocltv',
    'NUM_BO': 'num_bo',
    'DTI': 'dti',
    'FIRST_FLAG': 'FTHB_FLG',
    'PURPOSE': 'purpose',
    'PROP': 'PROP_TYP',
    'NO_UNITS': 'NUM_UNIT',
    'OCC_STAT': 'occ_stat',
    'STATE': 'state',
    'ZIP': 'zip_3',
    'MI_PCT': 'mi_pct',
    'PRODUCT': 'prod_type',
    'RELOCATION_MORTGAGE_INDICATOR': 'relo_flg'
})

# Find the latest ACT_PERIOD for each LOAN_ID
latest_periods = acquisitionFile.groupby('LOAN_ID')['ACT_PERIOD'].max().reset_index()
latest_periods = latest_periods.rename(columns={'ACT_PERIOD': 'first_period'})

# Merge with original acquisition data
acqFirstPeriod = pd.merge(
    latest_periods,
    acquisitionFile,
    left_on=['LOAN_ID', 'first_period'],
    right_on=['LOAN_ID', 'ACT_PERIOD'],
    how='left'
)

# Select final columns
acqFirstPeriod = acqFirstPeriod[[
    'LOAN_ID', 'ORIG_CHN', 'SELLER', 'orig_rt', 'orig_amt',
    'orig_trm', 'orig_date', 'first_pay', 'oltv',
    'ocltv', 'num_bo', 'dti', 'CSCORE_B', 'CSCORE_C',
    'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat',
    'state', 'zip_3', 'mi_pct', 'prod_type', 'MI_TYPE',
    'relo_flg', 'acquisition_date']]

# Select and rename columns for performance variables
performanceFile = combined_df[[
    'LOAN_ID', 'ACT_PERIOD', 'SERVICER', 'CURR_RATE', 'CURRENT_UPB',
    'LOAN_AGE', 'REM_MONTHS', 'ADJ_REM_MONTHS', 'MATR_DT', 'MSA',
    'DLQ_STATUS', 'MOD_FLAG', 'Zero_Bal_Code', 'ZB_DTE', 'LAST_PAID_INSTALLMENT_DATE',
    'FORECLOSURE_DATE', 'DISPOSITION_DATE', 'NON_INTEREST_BEARING_UPB', 'PRINCIPAL_FORGIVENESS_AMOUNT', 'LAST_UPB'
]].rename(columns={
    'ACT_PERIOD': 'period',
    'SERVICER': 'servicer',
    'CURR_RATE': 'curr_rte',
    'CURRENT_UPB': 'act_upb',
    'LOAN_AGE': 'loan_age',
    'REM_MONTHS': 'rem_mths',
    'ADJ_REM_MONTHS': 'adj_rem_months',
    'MATR_DT': 'maturity_date',
    'MSA': 'msa',
    'DLQ_STATUS': 'dlq_status',
    'MOD_FLAG': 'mod_ind',
    'Zero_Bal_Code': 'z_zb_code',
    'ZB_DTE': 'zb_date',
    'LAST_PAID_INSTALLMENT_DATE': 'lpi_dte',
    'FORECLOSURE_DATE': 'fcc_dte',
    'DISPOSITION_DATE': 'disp_dte',
    'NON_INTEREST_BEARING_UPB': 'non_int_upb',
    'PRINCIPAL_FORGIVENESS_AMOUNT': 'prin_forg_upb',
    'LAST_UPB': 'zb_upb'
})

def mmyyyy_to_date(val):
    if pd.notna(val) and val != '':
        # Clean up float-like strings: '92024.0' → '92024'
        val_str = str(val).split('.')[0]

        # Pad to 6 digits: '92024' → '092024'
        val_str = val_str.zfill(6)

        mm = val_str[:2]
        yyyy = val_str[2:]

        # Validate month and year
        if not (mm.isdigit() and yyyy.isdigit()):
            return pd.NA
        if int(mm) < 1 or int(mm) > 12:
            return pd.NA

        return f"{yyyy}-{mm}-01"
    return pd.NA

# Apply to all relevant columns
for col in ['maturity_date', 'zb_date', 'lpi_dte', 'fcc_dte', 'disp_dte']:
    performanceFile[col] = performanceFile[col].apply(mmyyyy_to_date)

# Start with a copy of the original DataFrame
baseTable1 = acqFirstPeriod.copy()

# 1. Rename 'acquisition_date' to 'AQSN_DTE'
baseTable1['AQSN_DTE'] = baseTable1['acquisition_date']

# 2. Recode MI_TYPE values
baseTable1['MI_TYPE'] = baseTable1['MI_TYPE'].map({
    '1': 'BPMI',
    '2': 'LPMI',
    '3': 'IPMI'
}).fillna('None')

# 3. Replace missing ocltv with oltv
baseTable1['ocltv'] = np.where(baseTable1['ocltv'].isna(), baseTable1['oltv'], baseTable1['ocltv'])

last_act_dte_table = performanceFile.groupby('LOAN_ID', as_index=False)['period'].max()
last_act_dte_table.rename(columns={'period': 'LAST_ACTIVITY_DATE'}, inplace=True)

# Step 1: Get the rows with the latest period per LOAN_ID
latest_rows = performanceFile.loc[
    performanceFile.groupby('LOAN_ID')['period'].idxmax()
]

# Step 2: Create LAST_UPB using zb_upb if not null, else act_upb
latest_rows['LAST_UPB'] = latest_rows['zb_upb'].combine_first(latest_rows['act_upb'])

# Step 3: Keep only needed columns
last_upb_table = latest_rows[['LOAN_ID', 'LAST_UPB']]

# Step 1: Filter non-null curr_rte
non_null_rt = performanceFile[performanceFile['curr_rte'].notna()]

# Step 2: Get the latest rate date per LOAN_ID
latest_rt_dates = non_null_rt.groupby('LOAN_ID', as_index=False)['period'].max()
latest_rt_dates.rename(columns={'period': 'LAST_RT_DATE'}, inplace=True)

# Step 3: Merge with performanceFile to get curr_rte at that date
last_rt_table = latest_rt_dates.merge(
    performanceFile,
    left_on=['LOAN_ID', 'LAST_RT_DATE'],
    right_on=['LOAN_ID', 'period'],
    how='left'
)

# Step 4: Select and rename
last_rt_table = last_rt_table[['LOAN_ID', 'curr_rte']]
last_rt_table.rename(columns={'curr_rte': 'LAST_RT'}, inplace=True)

# Step 5: Round LAST_RT to 3 decimals
last_rt_table['LAST_RT'] = last_rt_table['LAST_RT'].round(3)

# Step 1: Filter rows where z_zb_code is not an empty string
non_empty_zb = performanceFile[performanceFile['z_zb_code'] != '']

# Step 2: Get the latest period for each LOAN_ID
latest_zb_code = non_empty_zb.groupby('LOAN_ID', as_index=False)['period'].max()
latest_zb_code.rename(columns={'period': 'zb_code_dt'}, inplace=True)

# Step 3: Merge to get corresponding z_zb_code from performanceFile
zb_code_table = latest_zb_code.merge(
    performanceFile,
    left_on=['LOAN_ID', 'zb_code_dt'],
    right_on=['LOAN_ID', 'period'],
    how='left'
)

# Step 4: Select and rename columns
zb_code_table = zb_code_table[['LOAN_ID', 'z_zb_code']]
zb_code_table.rename(columns={'z_zb_code': 'zb_code'}, inplace=True)

# Step 1: Merge last activity date with performance data
max_table = last_act_dte_table.merge(
    performanceFile,
    left_on=['LOAN_ID', 'LAST_ACTIVITY_DATE'],
    right_on=['LOAN_ID', 'period'],
    how='left'
)

# Step 2: Merge with last UPB
max_table = max_table.merge(last_upb_table, on='LOAN_ID', how='left')

# Step 3: Merge with last rate
max_table = max_table.merge(last_rt_table, on='LOAN_ID', how='left')

# Step 4: Merge with ZB code table
max_table = max_table.merge(zb_code_table, on='LOAN_ID', how='left')

# Step 1: Filter rows with non-empty servicer
non_empty_servicer = performanceFile[performanceFile['servicer'] != '']

# Step 2: Get latest period per LOAN_ID
latest_servicer = non_empty_servicer.groupby('LOAN_ID', as_index=False)['period'].max()
latest_servicer.rename(columns={'period': 'servicer_period'}, inplace=True)

# Step 3: Merge back to get the servicer at that period
servicer_table = latest_servicer.merge(
    performanceFile,
    left_on=['LOAN_ID', 'servicer_period'],
    right_on=['LOAN_ID', 'period'],
    how='left'
)

# Step 4: Select and rename
servicer_table = servicer_table[['LOAN_ID', 'servicer']].rename(columns={'servicer': 'SERVICER'})

# Step 1: Sort by LOAN_ID and period (or assume current order is correct)
performanceFile_sorted = performanceFile.sort_values(by=['LOAN_ID', 'period'])

# Step 2: Get the second-to-last row per LOAN_ID
second_last_rows = performanceFile_sorted.groupby('LOAN_ID').nth(-2).reset_index()

# Step 3: Select and rename the column
non_int_upb_table = second_last_rows[['LOAN_ID', 'non_int_upb']].copy()
non_int_upb_table.rename(columns={'non_int_upb': 'NON_INT_UPB'}, inplace=True)

# Step 4: Replace NA with 0
non_int_upb_table['NON_INT_UPB'] = non_int_upb_table['NON_INT_UPB'].fillna(0)

# Start with baseTable1
baseTable2 = baseTable1.merge(max_table, on='LOAN_ID', how='left')
baseTable2 = baseTable2.merge(servicer_table, on='LOAN_ID', how='left')
baseTable2 = baseTable2.merge(non_int_upb_table, on='LOAN_ID', how='left')

fcc_table = performanceFile[
    performanceFile['lpi_dte'].notna() &
    performanceFile['fcc_dte'].notna() &
    performanceFile['disp_dte'].notna()
]

fcc_table = fcc_table.groupby('LOAN_ID', as_index=False).agg({
    'lpi_dte': 'max',
    'fcc_dte': 'max',
    'disp_dte': 'max'
})

# Rename columns to match R output
fcc_table.rename(columns={
    'lpi_dte': 'LPI_DTE',
    'fcc_dte': 'FCC_DTE',
    'disp_dte': 'DISP_DTE'
}, inplace=True)

baseTable3 = baseTable2.merge(fcc_table, on='LOAN_ID', how='left')

# Select columns
slimPerformanceFile = performanceFile[
    ['LOAN_ID', 'period', 'dlq_status', 'z_zb_code', 'act_upb', 'zb_upb', 'mod_ind', 'maturity_date', 'rem_mths']
].copy()

# Replace 'XX' with '999' in dlq_status, then convert to numeric (coerce errors)
slimPerformanceFile['dlq_status'] = slimPerformanceFile['dlq_status'].replace('XX', '999')
slimPerformanceFile['dlq_status'] = pd.to_numeric(slimPerformanceFile['dlq_status'], errors='coerce')

def create_f_table(df, dlq_min, dlq_label):
    filtered = df[
        (df['dlq_status'] >= dlq_min) &
        (df['dlq_status'] < 999) &
        (df['z_zb_code'].isna())
    ]
    min_period = filtered.groupby('LOAN_ID', as_index=False)['period'].min()
    min_period.rename(columns={'period': f'{dlq_label}_DTE'}, inplace=True)

    merged = min_period.merge(
        df,
        left_on=['LOAN_ID', f'{dlq_label}_DTE'],
        right_on=['LOAN_ID', 'period'],
        how='left'
    )

    return merged[['LOAN_ID', f'{dlq_label}_DTE', 'act_upb']].rename(columns={'act_upb': f'{dlq_label}_UPB'})

f30_table = create_f_table(slimPerformanceFile, 1, 'F30')
f60_table = create_f_table(slimPerformanceFile, 2, 'F60')
f90_table = create_f_table(slimPerformanceFile, 3, 'F90')
f120_table = create_f_table(slimPerformanceFile, 4, 'F120')
f180_table = create_f_table(slimPerformanceFile, 6, 'F180')

# Step 1: Filter rows with specific z_zb_code OR dlq_status condition
fce_filtered = slimPerformanceFile[
    (slimPerformanceFile['z_zb_code'].isin(['02', '03', '09', '15'])) |
    ((slimPerformanceFile['dlq_status'] >= 6) & (slimPerformanceFile['dlq_status'] < 999))
]

# Step 2: Find earliest period per LOAN_ID
fce_min_period = fce_filtered.groupby('LOAN_ID', as_index=False)['period'].min()
fce_min_period.rename(columns={'period': 'FCE_DTE'}, inplace=True)

# Step 3: Merge to get act_upb and zb_upb for FCE_DTE
fce_table = fce_min_period.merge(
    slimPerformanceFile,
    left_on=['LOAN_ID', 'FCE_DTE'],
    right_on=['LOAN_ID', 'period'],
    how='left'
)

# Step 4: Calculate FCE_UPB = zb_upb + act_upb
fce_table['FCE_UPB'] = fce_table['zb_upb'].fillna(0) + fce_table['act_upb'].fillna(0)

# Step 5: Select relevant columns
fce_table = fce_table[['LOAN_ID', 'FCE_DTE', 'FCE_UPB']]

fmod_filtered = slimPerformanceFile[
    (slimPerformanceFile['mod_ind'] == 'Y') &
    (slimPerformanceFile['z_zb_code'].isna())
]

fmod_dte_table = fmod_filtered.groupby('LOAN_ID', as_index=False)['period'].min()
fmod_dte_table.rename(columns={'period': 'FMOD_DTE'}, inplace=True)

# Step 1: Filter rows where mod_ind == 'Y' and z_zb_code == ''
fmod_filtered = slimPerformanceFile[
    (slimPerformanceFile['mod_ind'] == 'Y') &
    (slimPerformanceFile['z_zb_code'].isna())
]

# Step 2: Join with fmod_dte_table to get FMOD_DTE
fmod_merged = fmod_filtered.merge(fmod_dte_table, on='LOAN_ID', how='left')

# Helper function to convert period like 'YYYY-MM' or 'YYYY-MM-DD' to months count
def period_to_months(period):
    return period.year * 12 + period.month

# Step 3: Apply the date comparison condition:
# ((period in months) <= (FMOD_DTE in months) + 3)
fmod_merged['period_months'] = fmod_merged['period'].apply(period_to_months)
fmod_merged['FMOD_DTE_months'] = fmod_merged['FMOD_DTE'].apply(period_to_months)

# Ensure these columns are datetime
fmod_merged['period'] = pd.to_datetime(fmod_merged['period'])
fmod_merged['FMOD_DTE'] = pd.to_datetime(fmod_merged['FMOD_DTE'])

# Filter rows where period <= FMOD_DTE + 3 months
filtered_fmod = fmod_merged[
    fmod_merged['period'] <= (fmod_merged['FMOD_DTE'] + pd.DateOffset(months=3))
]

# Step 4: Group by LOAN_ID and get max act_upb as FMOD_UPB
fmod_summarized = filtered_fmod.groupby('LOAN_ID', as_index=False)['act_upb'].max()
fmod_summarized.rename(columns={'act_upb': 'FMOD_UPB'}, inplace=True)

# Step 5: Join back with fmod_dte_table and then with slimPerformanceFile on FMOD_DTE to get maturity_date
fmod_table = fmod_summarized.merge(fmod_dte_table, on='LOAN_ID', how='left')

fmod_table = fmod_table.merge(
    slimPerformanceFile[['LOAN_ID', 'period', 'maturity_date']],
    left_on=['LOAN_ID', 'FMOD_DTE'],
    right_on=['LOAN_ID', 'period'],
    how='left'
)

# Step 6: Select relevant columns
fmod_table = fmod_table[['LOAN_ID', 'FMOD_DTE', 'FMOD_UPB', 'maturity_date']]

# Step 1: Merge F120 table with acquisition file
num_120_table = f120_table.merge(acqFirstPeriod[['LOAN_ID', 'first_pay']], on='LOAN_ID', how='left')

# Step 2: Convert F120_DTE and FRST_DTE to datetime
num_120_table['F120_DTE'] = pd.to_datetime(num_120_table['F120_DTE'])
num_120_table['first_pay'] = pd.to_datetime(num_120_table['first_pay'])

# Step 3: Convert dates to total months (year * 12 + month)
num_120_table['z_num_periods_120'] = (
    (num_120_table['F120_DTE'].dt.year * 12 + num_120_table['F120_DTE'].dt.month) -
    (num_120_table['first_pay'].dt.year * 12 + num_120_table['first_pay'].dt.month) + 1
)

# Step 4: Select final columns
num_120_table = num_120_table[['LOAN_ID', 'z_num_periods_120']]

# Step 1: Filter rows where maturity_date is not missing
maturity_filtered = slimPerformanceFile[slimPerformanceFile['maturity_date'].notna()]

# Step 2: Get the earliest period (min) with a reported maturity_date per loan
maturity_date_period = maturity_filtered.groupby('LOAN_ID', as_index=False)['period'].min()
maturity_date_period.rename(columns={'period': 'maturity_date_period'}, inplace=True)

# Step 3: Merge to get the original maturity_date at that earliest period
orig_maturity_table = maturity_date_period.merge(
    slimPerformanceFile[['LOAN_ID', 'period', 'maturity_date']],
    left_on=['LOAN_ID', 'maturity_date_period'],
    right_on=['LOAN_ID', 'period'],
    how='left'
)

# Step 4: Select and rename
orig_maturity_table = orig_maturity_table[['LOAN_ID', 'maturity_date']].rename(columns={'maturity_date': 'orig_maturity_date'})

# Step 1: Sort by LOAN_ID and period
slimPerformanceFile_sorted = slimPerformanceFile.sort_values(by=['LOAN_ID', 'period'])

# Step 2: Group by LOAN_ID and calculate lag, change, and flag
slimPerformanceFile_sorted['prev_rem_mths'] = slimPerformanceFile_sorted.groupby('LOAN_ID')['rem_mths'].shift(1)
slimPerformanceFile_sorted['trm_chng'] = slimPerformanceFile_sorted['rem_mths'] - slimPerformanceFile_sorted['prev_rem_mths']
slimPerformanceFile_sorted['did_trm_chng'] = (slimPerformanceFile_sorted['trm_chng'] >= 0).astype(int)

# Step 3: Filter rows where term changed (did_trm_chng == 1)
trm_change_candidates = slimPerformanceFile_sorted[slimPerformanceFile_sorted['did_trm_chng'] == 1]

# Step 4: Get the earliest such period per LOAN_ID
trm_chng_table = trm_change_candidates.groupby('LOAN_ID', as_index=False)['period'].min()
trm_chng_table.rename(columns={'period': 'trm_chng_dt'}, inplace=True)

# Step 1: Join fmod_table with orig_maturity_table and trm_chng_table
modtrm_table = fmod_table.merge(orig_maturity_table, on='LOAN_ID', how='left')
modtrm_table = modtrm_table.merge(trm_chng_table, on='LOAN_ID', how='left')

# Step 2: Compute MODTRM_CHNG flag
modtrm_table['MODTRM_CHNG'] = (
    (modtrm_table['maturity_date'] != modtrm_table['orig_maturity_date']) |
    (modtrm_table['trm_chng_dt'].notna())
).astype(int)

# Step 3: Select final columns
modtrm_table = modtrm_table[['LOAN_ID', 'MODTRM_CHNG']]

# Step 1: Merge slimPerformanceFile with fmod_table to get FMOD_DTE
pre_mod_merge = slimPerformanceFile.merge(fmod_table[['LOAN_ID', 'FMOD_DTE']], on='LOAN_ID', how='left')

# Ensure both 'period' and 'FMOD_DTE' are datetime
pre_mod_merge['period'] = pd.to_datetime(pre_mod_merge['period'])
pre_mod_merge['FMOD_DTE'] = pd.to_datetime(pre_mod_merge['FMOD_DTE'])

# Step 2: Filter rows where period < FMOD_DTE
pre_mod_filtered = pre_mod_merge[pre_mod_merge['period'] < pre_mod_merge['FMOD_DTE']]

# Step 3: Get the latest period before FMOD_DTE per loan
pre_mod_periods = pre_mod_filtered.groupby('LOAN_ID', as_index=False)['period'].max()
pre_mod_periods.rename(columns={'period': 'pre_mod_period'}, inplace=True)

# Step 4: Merge back to get act_upb at that period
pre_mod_upb_table = pre_mod_periods.merge(
    slimPerformanceFile[['LOAN_ID', 'period', 'act_upb']],
    left_on=['LOAN_ID', 'pre_mod_period'],
    right_on=['LOAN_ID', 'period'],
    how='left'
)

# Step 5: Rename act_upb to pre_mod_upb
pre_mod_upb_table.rename(columns={'act_upb': 'pre_mod_upb'}, inplace=True)
pre_mod_upb_table = pre_mod_upb_table[['LOAN_ID', 'pre_mod_upb']]

# Step 1: Merge fmod_table with pre_mod_upb_table on LOAN_ID
modupb_table = fmod_table.merge(pre_mod_upb_table, on='LOAN_ID', how='left')

# Step 2: Compute the MODUPB_CHNG flag
modupb_table['MODUPB_CHNG'] = (modupb_table['FMOD_UPB'] >= modupb_table['pre_mod_upb']).astype(int)

# Step 3: Select final columns
modupb_table = modupb_table[['LOAN_ID', 'MODUPB_CHNG']]

# Step 1: Join all tables to baseTable3
baseTable4 = baseTable3 \
    .merge(f30_table, on='LOAN_ID', how='left') \
    .merge(f60_table, on='LOAN_ID', how='left') \
    .merge(f90_table, on='LOAN_ID', how='left') \
    .merge(f120_table, on='LOAN_ID', how='left') \
    .merge(f180_table, on='LOAN_ID', how='left') \
    .merge(fce_table, on='LOAN_ID', how='left') \
    .merge(fmod_table, on='LOAN_ID', how='left') \
    .merge(num_120_table, on='LOAN_ID', how='left') \
    .merge(modtrm_table, on='LOAN_ID', how='left') \
    .merge(modupb_table, on='LOAN_ID', how='left')

# Step 2: Replace missing UPBs with orig_amt if DTE is present
baseTable4['F30_UPB'] = np.where(
    baseTable4['F30_UPB'].isna() & baseTable4['F30_DTE'].notna(),
    baseTable4['orig_amt'],
    baseTable4['F30_UPB']
)

baseTable4['F60_UPB'] = np.where(
    baseTable4['F60_UPB'].isna() & baseTable4['F60_DTE'].notna(),
    baseTable4['orig_amt'],
    baseTable4['F60_UPB']
)

baseTable4['F90_UPB'] = np.where(
    baseTable4['F90_UPB'].isna() & baseTable4['F90_DTE'].notna(),
    baseTable4['orig_amt'],
    baseTable4['F90_UPB']
)

baseTable4['F120_UPB'] = np.where(
    baseTable4['F120_UPB'].isna() & baseTable4['F120_DTE'].notna(),
    baseTable4['orig_amt'],
    baseTable4['F120_UPB']
)

baseTable4['F180_UPB'] = np.where(
    baseTable4['F180_UPB'].isna() & baseTable4['F180_DTE'].notna(),
    baseTable4['orig_amt'],
    baseTable4['F180_UPB']
)

baseTable4['FCE_UPB'] = np.where(
    baseTable4['FCE_UPB'].isna() & baseTable4['FCE_DTE'].notna(),
    baseTable4['orig_amt'],
    baseTable4['FCE_UPB']
)

# Ensure datetime columns are strings or datetime for slicing
baseTable5 = baseTable4.copy()

# LAST_DTE: if DISP_DTE present, use it; else LAST_ACTIVITY_DATE
baseTable5['LAST_DTE'] = np.where(baseTable5['disp_dte'].notna(), baseTable5['disp_dte'], baseTable5['LAST_ACTIVITY_DATE'])

# Binary flags
baseTable5['MOD_FLAG'] = baseTable5['FMOD_DTE'].notna().astype(int)

# Principal forgiveness cost
baseTable5['PFG_COST'] = baseTable5['prin_forg_upb']

# MODFG_COST logic
baseTable5['MODFG_COST'] = np.where(baseTable5['mod_ind'] == 'Y', 0.0, np.nan)
baseTable5['MODFG_COST'] = np.where(
    (baseTable5['mod_ind'] == 'Y') & (baseTable5['PFG_COST'] > 0),
    baseTable5['PFG_COST'],
    baseTable5['MODFG_COST']
)

# Fill missing MODTRM_CHNG, MODUPB_CHNG with 0
baseTable5['MODTRM_CHNG'] = baseTable5['MODTRM_CHNG'].fillna(0)
baseTable5['MODUPB_CHNG'] = baseTable5['MODUPB_CHNG'].fillna(0)

# CSCORE_MN logic
baseTable5['CSCORE_MN'] = np.where(
    (baseTable5['CSCORE_C'].notna()) & (baseTable5['CSCORE_C'] < baseTable5['CSCORE_B']),
    baseTable5['CSCORE_C'],
    baseTable5['CSCORE_B']
)
baseTable5['CSCORE_MN'] = baseTable5['CSCORE_MN'].combine_first(baseTable5['CSCORE_B'])
baseTable5['CSCORE_MN'] = baseTable5['CSCORE_MN'].combine_first(baseTable5['CSCORE_C'])

# ORIG_VAL = orig_amt / (oltv / 100)
baseTable5['ORIG_VAL'] = round(baseTable5['orig_amt'] / (baseTable5['oltv'] / 100), 2)

# dlq_status cleanup
baseTable5['dlq_status'] = baseTable5['dlq_status'].replace({'X': '999', 'XX': '999'})
baseTable5['z_last_status'] = pd.to_numeric(baseTable5['dlq_status'], errors='coerce')

def get_last_stat(row):
    zb_code = row['zb_code']
    z = row['z_last_status']

    # Safe check for zb_code
    if pd.notna(zb_code):
        if zb_code == '09': return 'F'
        if zb_code == '03': return 'S'
        if zb_code == '02': return 'T'
        if zb_code == '06': return 'R'
        if zb_code == '15': return 'N'
        if zb_code == '16': return 'L'
        if zb_code == '01': return 'P'

    # Safe numeric check for z
    z_num = pd.to_numeric(z, errors='coerce')
    if pd.notna(z_num):
        z_int = int(z_num)
        if 9 <= z_int < 999:
            return '9'
        elif 0 <= z_int <= 8:
            return str(z_int)

    return 'X'

baseTable5['LAST_STAT'] = baseTable5.apply(get_last_stat, axis=1)

# FCC_DTE fallback
baseTable5['FCC_DTE'] = np.where(
    baseTable5['FCC_DTE'].isna() & baseTable5['LAST_STAT'].isin(['F', 'S', 'N', 'T']),
    baseTable5['zb_date'],
    baseTable5['FCC_DTE']
)

# COMPLT_FLG
baseTable5['COMPLT_FLG'] = np.where(baseTable5['disp_dte'].notna(), 1, np.nan)
baseTable5['COMPLT_FLG'] = np.where(~baseTable5['LAST_STAT'].isin(['F', 'S', 'N', 'T']), np.nan, baseTable5['COMPLT_FLG'])

# INT_COST
def compute_int_cost(row):
    if row['COMPLT_FLG'] == 1 and pd.notna(row['LPI_DTE']) and pd.notna(row['LAST_DTE']):
        try:
            end = int(row['LAST_DTE'][:4]) * 12 + int(row['LAST_DTE'][5:7])
            start = int(row['LPI_DTE'][:4]) * 12 + int(row['LPI_DTE'][5:7])
            months_diff = end - start
            rate = ((row['LAST_RT'] / 100) - 0.0035) / 12
            return round(months_diff * rate * (row['LAST_UPB'] + (-1 * row['NON_INT_UPB'])), 2)
        except:
            return np.nan
    return np.nan

baseTable5['INT_COST'] = baseTable5.apply(compute_int_cost, axis=1)
baseTable5['INT_COST'] = np.where(
    (baseTable5['COMPLT_FLG'] == 1) & (baseTable5['INT_COST'].isna()),
    0,
    baseTable5['INT_COST']
)

# Zero-out other costs where COMPLT_FLG == 1
baseTable5['PFG_COST'] = np.where(
    (baseTable5['COMPLT_FLG'] == 1) & (baseTable5[col].isna()),
    0,
    baseTable5['PFG_COST']
)

# NET_LOSS
baseTable5['NET_LOSS'] = np.where(
    baseTable5['COMPLT_FLG'] == 1,
    round(+ baseTable5['PFG_COST'], 2), np.nan)

# NET_SEV
baseTable5['NET_SEV'] = np.where(
    baseTable5['COMPLT_FLG'] == 1,
    round(baseTable5['NET_LOSS'] / baseTable5['LAST_UPB'], 6),
    np.nan
)

# Merge baseTable1 and performanceFile on 'LOAN_ID'
modir_table = baseTable1.merge(performanceFile, on='LOAN_ID', how='left')

# Filter only modified loans
modir_table = modir_table[modir_table['mod_ind'] == 'Y'].copy()

# Fill missing non_int_upb with 0
modir_table['non_int_upb'] = modir_table['non_int_upb'].fillna(0)

# Compute MODIR_COST and MODFB_COST
modir_table['modir_cost'] = round(
    ((modir_table['orig_rt'] - modir_table['curr_rte']) / 1200) * modir_table['act_upb'], 2
)
modir_table['modfb_cost'] = round(
    (modir_table['curr_rte'] / 1200) * modir_table['non_int_upb'].where(modir_table['non_int_upb'] > 0, 0), 2
)

# Group and summarize
modir_summary = modir_table.groupby('LOAN_ID', as_index=False).agg({
    'modir_cost': 'sum',
    'modfb_cost': 'sum'
})

# Rename columns
modir_summary.rename(columns={
    'modir_cost': 'MODIR_COST',
    'modfb_cost': 'MODFB_COST'
}, inplace=True)

# Compute MODTOT_COST
modir_summary['MODTOT_COST'] = round(modir_summary['MODIR_COST'] + modir_summary['MODFB_COST'], 2)

# Merge baseTable5 with modir_summary (modir_table in R)
baseTable6 = baseTable5.merge(modir_summary, on='LOAN_ID', how='left')

# Fill COMPLT_FLG as string and handle missing values
baseTable6['COMPLT_FLG'] = baseTable6['COMPLT_FLG'].astype(str)
baseTable6['COMPLT_FLG'] = baseTable6['COMPLT_FLG'].fillna('')


# Replace non_int_upb with 0 if COMPLT_FLG == '1' and value is NaN
baseTable6['non_int_upb'] = baseTable6.apply(
    lambda row: 0 if row['COMPLT_FLG'] == '1' and pd.isna(row['non_int_upb']) else row['non_int_upb'],
    axis=1
)

# Ensure orig_rt is numeric and rounded
baseTable6['orig_rt'] = pd.to_numeric(baseTable6['orig_rt'], errors='coerce').round(3)

# Define a helper to convert YYYY-MM format to total months
def to_total_months(date_str):
    try:
        y, m = int(date_str[:4]), int(date_str[5:7])
        return y * 12 + m
    except:
        return pd.NA

# Apply month conversions
baseTable6['LAST_DTE_months'] = baseTable6['LAST_DTE'].apply(to_total_months)
baseTable6['zb_date_months'] = baseTable6['zb_date'].apply(to_total_months)

# Adjust MODIR_COST
baseTable6['MODIR_COST'] = baseTable6.apply(
    lambda row: round(
        row['MODIR_COST'] + 
        ((row['LAST_DTE_months'] - row['zb_date_months']) * ((row['orig_rt'] - row['LAST_RT']) / 1200) * row['LAST_UPB']),
        2
    ) if row['COMPLT_FLG'] == '1' and pd.notna(row['MODIR_COST']) else row['MODIR_COST'],
    axis=1
)

# Adjust MODFB_COST
baseTable6['MODFB_COST'] = baseTable6.apply(
    lambda row: round(
        row['MODFB_COST'] + 
        ((row['LAST_DTE_months'] - row['zb_date_months']) * (row['LAST_RT'] / 1200) * row['non_int_upb']),
        2
    ) if row['COMPLT_FLG'] == '1' and pd.notna(row['MODFB_COST']) else row['MODFB_COST'],
    axis=1
)

# Convert COMPLT_FLG back to numeric
baseTable6['COMPLT_FLG'] = pd.to_numeric(baseTable6['COMPLT_FLG'], errors='coerce')

columns_to_select = [
    'LOAN_ID', 'ORIG_CHN', 'SELLER', 'orig_rt', 'orig_amt',
    'orig_trm', 'oltv', 'ocltv', 'num_bo', 'dti',
    'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT',
    'occ_stat', 'state', 'zip_3', 'mi_pct', 'CSCORE_C',
    'relo_flg', 'MI_TYPE', 'AQSN_DTE', 'orig_date', 'first_pay',
    'LAST_RT', 'LAST_UPB', 'msa', 'LAST_ACTIVITY_DATE',
    'LPI_DTE', 'FCC_DTE', 'DISP_DTE', 'SERVICER', 'F30_DTE',
    'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE',
    'F180_UPB', 'FCE_UPB', 'F30_UPB', 'F60_UPB', 'F90_UPB',
    'MOD_FLAG', 'FMOD_DTE', 'FMOD_UPB', 'MODIR_COST', 'MODFB_COST',
    'MODFG_COST', 'MODTRM_CHNG', 'MODUPB_CHNG', 'z_num_periods_120', 'F120_UPB',
    'CSCORE_MN', 'ORIG_VAL', 'LAST_DTE', 'LAST_STAT', 'COMPLT_FLG',
    'INT_COST', 'PFG_COST', 'NET_LOSS', 'NET_SEV', 'MODTOT_COST'
]

# Create baseTable7 with only selected and reordered columns
baseTable7 = baseTable6[columns_to_select]

baseTable7.to_csv("Clean_Data.csv", index=False)





























