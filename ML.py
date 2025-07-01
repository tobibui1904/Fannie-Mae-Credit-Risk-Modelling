import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
from sklearn.utils import resample
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.optim import ClippedAdam
from pyro.infer import TraceMeanField_ELBO
from sklearn.metrics import classification_report, confusion_matrix
from pyro.infer import SVI, Trace_ELBO

df = pd.read_csv(r"C:\Users\buitu\Coding Projects\PD\Clean_Data.csv", low_memory=False)
all_nan_cols = df.columns[df.isna().all()]
df = df.drop(columns=all_nan_cols, axis= 1)
nan_cols = df.columns[df.isna().any()]
df['F30_DTE'] = df['F30_DTE'].fillna(df['F30_DTE'].mode()[0])
df['F60_DTE'] = df['F60_DTE'].fillna(df['F60_DTE'].mode()[0])
df['F90_DTE'] = df['F90_DTE'].fillna(df['F90_DTE'].mode()[0])
df['F120_DTE'] = df['F120_DTE'].fillna(df['F120_DTE'].mode()[0])
df['F180_DTE'] = df['F180_DTE'].fillna(df['F180_DTE'].mode()[0])
df['FCE_DTE'] = df['FCE_DTE'].fillna(df['FCE_DTE'].mode()[0])
df['FMOD_DTE'] = df['FMOD_DTE'].fillna(df['FMOD_DTE'].mode()[0])
df['F180_UPB'] = df['F180_UPB'].fillna(df['F180_UPB'].mean())
df['FCE_UPB'] = df['FCE_UPB'].fillna(df['FCE_UPB'].mean())
df['F30_UPB'] = df['F30_UPB'].fillna(df['F30_UPB'].mean())
df['F60_UPB'] = df['F60_UPB'].fillna(df['F60_UPB'].mean())
df['F90_UPB'] = df['F90_UPB'].fillna(df['F90_UPB'].mean())
df['FMOD_UPB'] = df['FMOD_UPB'].fillna(df['FMOD_UPB'].mean())
df['MODIR_COST'] = df['MODIR_COST'].fillna(df['MODIR_COST'].mean())
df['MODFB_COST'] = df['MODFB_COST'].fillna(df['MODFB_COST'].mean())
df['MODFG_COST'] = df['MODFG_COST'].fillna(df['MODFG_COST'].mean())
df['z_num_periods_120'] = df['z_num_periods_120'].fillna(df['z_num_periods_120'].mean())
df['F120_UPB'] = df['F120_UPB'].fillna(df['F120_UPB'].mean())
df['MODTOT_COST'] = df['MODTOT_COST'].fillna(df['MODTOT_COST'].mean())

leaky = ['LAST_RT', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'LAST_DTE', 'LPI_DTE', 'FCC_DTE', 'DISP_DTE', 'F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE', 'F180_UPB', 'FCE_UPB', 'F30_UPB', 'F60_UPB', 'F90_UPB', 'FMOD_DTE', 'FMOD_UPB', 'MODIR_COST', 'MODFB_COST', 'MODFG_COST', 'MODTRM_CHNG', 'MODUPB_CHNG', 'z_num_periods_120', 'F120_UPB', 'LAST_DTE', 'PFG_COST', 'MODTOT_COST']
intact_df = df.copy()
intact_df = intact_df[intact_df['MOD_FLAG'] == 0]

intact_df = intact_df.drop(columns= leaky, axis= 1)
intact_df['log_orig_amt'] = np.log(intact_df['orig_amt'])
def bucket_orig_term(term):
    if term <= 180:
        return 0
    elif 181 <= term <= 240:
        return 1
    else:
        return 2

intact_df['orig_term_bucket'] = intact_df['orig_trm'].apply(bucket_orig_term)

def bucket_orig_amt(term):
    if term < 50000:
        return 0
    elif 50000 <= term <= 100000:
        return 1
    elif 100001 <= term <= 200000:
        return 2
    elif 200001 <= term <= 400000:
        return 3
    else:
        return 4

intact_df['orig_amt_bucket'] = intact_df['orig_amt'].apply(bucket_orig_amt)

def bucket_oltv(term):
    if term < 60:
        return 0
    elif 60 <= term <= 80:
        return 1
    elif 81 <= term <= 90:
        return 2
    else:
        return 3

intact_df['oltv_bucket'] = intact_df['oltv'].apply(bucket_oltv)

def bucket_ocltv(term):
    if term < 80:
        return 0
    elif 80 <= term <= 90:
        return 1
    elif 91 <= term <= 95:
        return 2
    elif 96 <= term <= 100:
        return 3
    else:
        return 4

intact_df['ocltv_bucket'] = intact_df['ocltv'].apply(bucket_ocltv)

def bucket_dti(term):
    if term < 20:
        return 0
    elif 20 <= term <= 35:
        return 1
    elif 36 <= term <= 43:
        return 2
    elif 44 <= term <= 50:
        return 3
    else:
        return 4

intact_df['dti_bucket'] = intact_df['dti'].apply(bucket_dti)

def bucket_mi_pct(term):
    if 6 <= term <= 12:
        return 0
    elif 12.01 <= term <= 25:
        return 1
    else:
        return 2

intact_df['mi_pct_bucket'] = intact_df['mi_pct'].apply(bucket_mi_pct)

def bucket_orig_val(term):
    if term < 100000:
        return 0
    elif 100000 <= term <= 200000:
        return 1
    elif 200001 <= term <= 400000:
        return 2
    elif 400001 <= term <= 700000:
        return 3
    elif 700001 <= term <= 1000000:
        return 4
    else:
        return 5

intact_df['orig_val_bucket'] = intact_df['ORIG_VAL'].apply(bucket_orig_val)

def credit_score_bucket(score):
    if score >= 800:
        return 4
    elif 740 <= score <= 799:
        return 3
    elif 670 <= score <= 739:
        return 2
    elif 580 <= score <= 669:
        return 1
    else:
        return 0

# Apply to borrower credit score column (adjust the column name if needed)
intact_df['CSCORE_B_bucket'] = intact_df['CSCORE_B'].apply(credit_score_bucket)
intact_df['CSCORE_C_bucket'] = intact_df['CSCORE_C'].apply(credit_score_bucket)
intact_df['CSCORE_MN_bucket'] = intact_df['CSCORE_MN'].apply(credit_score_bucket)

# List of bucket columns to analyze
bucket_vars = ['orig_term_bucket', 'orig_amt_bucket', 'oltv_bucket', 'ocltv_bucket', 'dti_bucket', 'mi_pct_bucket', 'orig_val_bucket', 'CSCORE_MN_bucket']

# Dictionary to hold WOE/IV results per variable
woe_results = {}

# Loop through each bucket variable
for var in tqdm(bucket_vars):
    woe_df = (
        intact_df.groupby(var)['LAST_STAT']
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Handle columns as strings or ints
    stat_cols_1_to_9 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    if all(isinstance(col, str) for col in woe_df.columns[1:]):
        stat_cols_1_to_9 = [str(c) for c in stat_cols_1_to_9]
        stat_col_0 = '0'
    else:
        stat_col_0 = 0

    # Create summarized WOE dataframe
    woe_sum_df = woe_df[[var]].copy()
    woe_sum_df['last_stat_0'] = woe_df[stat_col_0]
    woe_sum_df['last_stat_1_to_9_sum'] = woe_df[stat_cols_1_to_9].sum(axis=1)

    # Calculate distributions
    total_good = woe_sum_df['last_stat_0'].sum()
    total_bad = woe_sum_df['last_stat_1_to_9_sum'].sum()

    woe_sum_df['dist_good'] = woe_sum_df['last_stat_0'] / total_good
    woe_sum_df['dist_bad'] = woe_sum_df['last_stat_1_to_9_sum'] / total_bad

    # Compute WOE
    woe_sum_df['WOE'] = np.log(
        (woe_sum_df['dist_good'].replace(0, 1e-6)) /
        (woe_sum_df['dist_bad'].replace(0, 1e-6))
    )

    # Compute IV
    woe_sum_df['IV'] = (woe_sum_df['dist_good'] - woe_sum_df['dist_bad']) * woe_sum_df['WOE']
    iv_total = woe_sum_df['IV'].sum()

    # Store in dictionary
    woe_results[var] = {
        'woe_df': woe_sum_df,
        'IV_total': iv_total
    }

interaction_pairs = list(combinations(bucket_vars, 2))
interaction_woe_results = []

for var1, var2 in tqdm(interaction_pairs):
    col_name = f"{var1}_x_{var2}"
    intact_df[col_name] = intact_df[var1].astype(str) + '_' + intact_df[var2].astype(str)

    woe_df = (
        intact_df.groupby(col_name)['LAST_STAT']
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Detect data type
    stat_cols_1_to_9 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    if all(isinstance(col, str) for col in woe_df.columns[1:]):
        stat_cols_1_to_9 = [str(c) for c in stat_cols_1_to_9]
        stat_col_0 = '0'
    else:
        stat_col_0 = 0

    # Aggregate
    woe_sum_df = woe_df[[col_name]].copy()
    woe_sum_df['last_stat_0'] = woe_df[stat_col_0]
    woe_sum_df['last_stat_1_to_9_sum'] = woe_df[stat_cols_1_to_9].sum(axis=1)

    total_good = woe_sum_df['last_stat_0'].sum()
    total_bad = woe_sum_df['last_stat_1_to_9_sum'].sum()

    woe_sum_df['dist_good'] = woe_sum_df['last_stat_0'] / total_good
    woe_sum_df['dist_bad'] = woe_sum_df['last_stat_1_to_9_sum'] / total_bad

    woe_sum_df['WOE'] = np.log(
        (woe_sum_df['dist_good'].replace(0, 1e-6)) /
        (woe_sum_df['dist_bad'].replace(0, 1e-6))
    )

    woe_sum_df['IV'] = (woe_sum_df['dist_good'] - woe_sum_df['dist_bad']) * woe_sum_df['WOE']
    iv_total = woe_sum_df['IV'].sum()

    interaction_woe_results.append({
        'Interaction': col_name,
        'Var1': var1,
        'Var2': var2,
        'IV': iv_total
    })

interaction_iv_df = pd.DataFrame(interaction_woe_results)
strong_interactions = interaction_iv_df.sort_values(by='IV', ascending=False)

intact_df['dti_CSCORE_MN'] = intact_df['dti'] * intact_df['CSCORE_MN']
intact_df['oltv_CSCORE_MN'] = intact_df['oltv'] * intact_df['CSCORE_MN']
intact_df['ocltv_CSCORE_MN'] = intact_df['ocltv'] * intact_df['CSCORE_MN']
intact_df['orig_amt_CSCORE_MN'] = intact_df['orig_amt'] * intact_df['CSCORE_MN']
intact_df['orig_term_CSCORE_MN'] = intact_df['orig_trm'] * intact_df['CSCORE_MN']
intact_df['orig_val_CSCORE_MN'] = intact_df['ORIG_VAL'] * intact_df['CSCORE_MN']
intact_df['mi_pct_CSCORE_MN'] = intact_df['mi_pct'] * intact_df['CSCORE_MN']
intact_df['state_encoded'] = intact_df['state'].astype('category').cat.codes

features = [
    'dti_CSCORE_MN', 'oltv_CSCORE_MN', 'ocltv_CSCORE_MN',
    'orig_amt_CSCORE_MN', 'orig_term_CSCORE_MN',
    'orig_val_CSCORE_MN', 'mi_pct_CSCORE_MN', 'CSCORE_MN',
    'state_encoded', 'num_bo'
]

df_majority = intact_df[intact_df['LAST_STAT'] == 1]
df_minority = intact_df[intact_df['LAST_STAT'] == 0]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced[features].values
y = df_balanced['LAST_STAT'].apply(lambda x: 1 if x == 0 else 0).values
loan_ids = df_balanced['LOAN_ID'].values  # Make sure Loan_ID is present

class BayesianRegression(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = PyroModule[nn.Linear](input_dim, 16)
        self.linear1.weight = PyroSample(dist.Normal(0., 0.05).expand([16, input_dim]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., 0.05).expand([16]).to_event(1))

        self.linear2 = PyroModule[nn.Linear](16, 1)
        self.linear2.weight = PyroSample(dist.Normal(0., 0.05).expand([1, 16]).to_event(2))
        self.linear2.bias = PyroSample(dist.Normal(0., 0.05).expand([1]).to_event(1))

    def forward(self, x, y=None):
        x = torch.relu(self.linear1(x))
        mean = torch.sigmoid(self.linear2(x)).squeeze()
        with pyro.plate("data", x.shape[0]):
            return pyro.sample("obs", dist.Bernoulli(mean), obs=y)


def guide(x, y=None):
    for name, param in pyro.get_param_store().items():
        pyro.module(name, param)

    for m in pyro.get_param_store().keys():
        if m.endswith("weight_loc") or m.endswith("bias_loc"):
            pyro.param(m, pyro.get_param_store()[m])

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, loan_ids, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = BayesianRegression(input_dim=X_train.shape[1])
latent_dim = (X_train.shape[1] * 16) + 16 + (16 * 1) + 1
guide = AutoLowRankMultivariateNormal(model, rank=latent_dim // 5)  # rank = latent dim / 5 usually


svi = SVI(model, guide, ClippedAdam({"lr": 5e-5, "clip_norm": 10.0}), loss=Trace_ELBO())

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

num_steps = 20000
for step in range(num_steps):
    loss = svi.step(X_train_tensor, y_train_tensor)
    if step % 100 == 0:
        print(f"[{step}] ELBO Loss: {loss:.4f}")

elbo = TraceMeanField_ELBO()
loss = elbo.differentiable_loss(model, guide, X_train_tensor, y_train_tensor)
print(f"Final ELBO: {-loss.item():.4f}")

def predict_with_uncertainty(guide, x, n_samples=100):
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=n_samples, return_sites=["obs"])
    samples = predictive(x)
    pred_probs = samples["obs"].detach().numpy()
    return pred_probs.mean(axis=0), pred_probs.std(axis=0)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
mean_preds, std_preds = predict_with_uncertainty(guide, X_test_tensor)

# Create a DataFrame with predictions
pred_df = pd.DataFrame({
    'Loan_ID': id_test,
    'p_good_mean': mean_preds,
    'p_good_std': std_preds
})
threshold = 0.5  # Set a threshold for classification
pred_df['predicted_class'] = (pred_df['p_good_mean'] >= threshold).astype(int)

print(confusion_matrix(y_test, pred_df['predicted_class']))
print(classification_report(y_test, pred_df['predicted_class'], target_names=["Default", "Non-default"]))

print(pred_df.head())





