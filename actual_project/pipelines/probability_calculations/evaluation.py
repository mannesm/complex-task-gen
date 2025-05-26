import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.extend(
    [
        '/gpfs/home6/mmokkenstorm/tmp/complex_task_gen/',
        '/tmp/pycharm_project_977',
        '/home/mmokkenstorm/tmp/complex_task_gen/actual_project',
    ],
)


# from pipelines.gsm_evaluation_dataset_creation import create_full_gsm8k_test_dataset
from complex_task_gen.actual_project.pipelines.gsm_evaluation_dataset_creation import create_full_gsm8k_test_dataset

FILE_1 = '/Users/mannes/thesis/eval_full_20250525_182752Qwen/Qwen2.5-Math-7B-Instruct.json'
FILE_2 = '/Users/mannes/thesis/rows_step800_20250525_174102Qwen/Qwen2.5-Math-7B-Instruct.csv'

df_1 = pd.read_json(FILE_1)
df_2 = pd.read_csv(FILE_2)

df_total = pd.concat([df_1, df_2], ignore_index=True)
df_total.sort_values(by=[df_total.columns[0]], inplace=True)


OUTPUT_FOLDER_LOCATION = '/home/mmokkenstorm/tmp/complex_task_gen/output'

# Load generated & test data
# df_gen = pd.read_json(OUTPUT_FOLDER_LOCATION + '/gsm8k_pass1_logp_15b_sub.json')
test_df = create_full_gsm8k_test_dataset(to_df=True)
test_df.rename(columns={'original_id': 'id'}, inplace=True)

# 1) regex patterns in order of specificity → last match

df_eval = pd.merge(test_df, df_total, on='id')


def extract_numeric_value_hybrid(text: str) -> float | None:
    txt = str(text).replace(',', '').strip()

    # Step 1: High-confidence \boxed{}
    match = re.search(r'\\?boxed\{\$?(-?\d+(?:\.\d+)?)\}', txt)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Step 2: Try other patterns (same list as before)
    regex_patterns = [
        r'\bAnswer[:\-\s]+\$?(-?\d+(?:\.\d+)?)\b',
        r'\b(?:Therefore|Thus|Hence)[,:\s]+\$?(-?\d+(?:\.\d+)?)',
        r'\b(?:The (?:final\s+)?answer is)\s+\$?(-?\d+(?:\.\d+)?)',
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'=\s*(-?\d+(?:\.\d+)?)[\.\)]?\s*$',
        r'(-?\d+(?:\.\d+)?)\s*(?:dollars|km|miles|liters|hours|minutes|seconds)\b',
    ]

    for pat in regex_patterns:
        hits = re.findall(pat, txt, flags=re.IGNORECASE)
        if hits:
            try:
                return float(hits[-1])
            except ValueError:
                continue

    # Step 3: Fallback — last number in last sentence
    last_sentence = txt.rsplit('.', 1)[-1]
    nums = re.findall(r'\b(-?\d+(?:\.\d+)?)\b', last_sentence)
    if nums:
        return float(nums[-1])

    return None


# Merge and evaluate
# df_eval = pd.merge(test_df, df_gen, on='id')

# Extract numeric from the generated-answer string
df_eval['generated_numeric_answer_v2'] = df_eval['generated_answer'].apply(extract_numeric_value_hybrid)


df_sliced = df_eval[['id', 'generated_answer', 'generated_numeric_answer', 'numeric_answer']].copy()
# Compare to the true numeric_answer
df_eval['pass1_2'] = df_eval['numeric_answer'] == df_eval['generated_numeric_answer']

df_eval['pass1'].value_counts()
df_eval['pass1_2'].value_counts()

failures = df_eval.loc[~df_eval['pass1'], ['id', 'generated_answer', 'generated_numeric_answer', 'numeric_answer']]
print(f'\nNumber of fails: {len(failures)}')
print(failures.head(10).to_string(index=False))


# Build correlation table
def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pass1_int'] = df['pass1'].astype(int)
    rows = []
    for col in [
        'logp_sum',
        'avg_logp',
        'perplexity',
        'token_entropy',
        'prompt_len',
        'num_ops',
        'voting_entropy',
    ]:
        if df[col].nunique() <= 1 or df[col].isna().all():
            rows.append(
                {
                    'metric': col,
                    'pearson': np.nan,
                    'pearson_p': np.nan,
                    'spearman': np.nan,
                    'spearman_p': np.nan,
                },
            )
        else:
            pear, p_pear = pearsonr(df[col], df['pass1_int'])
            spear, p_spear = spearmanr(df[col], df['pass1_int'])
            rows.append(
                {
                    'metric': col,
                    'pearson': pear,
                    'pearson_p': p_pear,
                    'spearman': spear,
                    'spearman_p': p_spear,
                },
            )
    return pd.DataFrame(rows)


df_corr = corr_table(df_eval)
print('\nCorrelation table:')
print(df_corr.to_string(index=False))
df_eval['pass1_int'] = df_eval['pass1'].astype(int)
df_eval.to_csv('/Users/mannes/thesis/correlation_evaluation/Qwen7B_math_run.csv')
df_eval.columns

# ---------------------------------------------------------
# Visual analytics for GSM-8K pass-fail evaluation
# ---------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 0.  Global styling – tweak once, all figures inherit the settings
# ------------------------------------------------------------------
sns.set_theme(
    context='paper',  # a bit smaller than 'notebook'
    style='whitegrid',
    font='DejaVu Sans',
    rc={
        'figure.dpi': 300,  # high-res images for print
        'savefig.dpi': 300,
        'axes.titlesize': 'medium',
        'axes.labelsize': 'small',
        'legend.fontsize': 'x-small',
    },
)

# Add the binary column explicitly if you haven’t already
df_eval['pass1_int'] = df_eval['pass1'].astype(int)

num_cols = [
    'logp_sum',
    'avg_logp',
    'perplexity',
    'token_entropy',
    'prompt_len',
    'num_ops',
    'voting_entropy',
    'pass1_int',
]

# ------------------------------------------------------------------
# 1.  Correlation heat-map  (Spearman)
# ------------------------------------------------------------------
plt.figure(figsize=(4.5, 4))
corr = df_eval[num_cols].corr(method='spearman')
sns.heatmap(
    corr,
    cmap='vlag',
    center=0,
    linewidths=0.5,
    cbar_kws=dict(label='Spearman ρ'),
    square=True,
    annot=True,
    fmt='.2f',
)
plt.title('Spearman correlation matrix')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 2.  Bar chart of metric ⇄ accuracy correlations
# ------------------------------------------------------------------
plt.figure(figsize=(5, 2.5))
# Use the table we already computed
plot_df = df_corr.sort_values('spearman', key=lambda s: s.abs(), ascending=False).dropna(subset=['spearman'])
sns.barplot(
    data=plot_df,
    y='metric',
    x='spearman',
    palette='vlag',
    orient='h',
)
plt.axvline(0, color='k', lw=0.8)
plt.xlabel('Spearman ρ with pass/fail')
plt.ylabel('')
plt.title('Strength & direction of each metric')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3.  Distribution plots for every metric, split by outcome
# ------------------------------------------------------------------
n_cols = 3
n_rows = -(-len(num_cols[:-1]) // n_cols)  # ceil division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.4))
axes = axes.flatten()

for ax, col in zip(axes, num_cols[:-1], strict=False):  # skip pass1_int in dists
    sns.histplot(
        data=df_eval,
        x=col,
        hue='pass1',
        kde=True,
        stat='density',
        common_norm=False,
        element='step',
        palette=['#d94801', '#006d77'],
        ax=ax,
    )
    ax.set_title(col)
    ax.set_ylabel('Density')
    ax.legend_.remove()

# Hide empty subplots (if any)
for j in range(len(num_cols) - 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Metric distributions by correctness', y=1.02)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4.  Pair-plot of metrics with hue = pass/fail
#     (quick but often very informative)
# ------------------------------------------------------------------
sns.pairplot(
    data=df_eval,
    vars=[
        'avg_logp',
        'perplexity',
        'token_entropy',
        'num_ops',
        'voting_entropy',
    ],
    hue='pass1',
    diag_kind='kde',
    corner=True,  # lower-triangle only
    plot_kws=dict(alpha=0.5, rasterized=True),  # raster for big figs
)
plt.suptitle('Pair-wise relationships (pass vs. fail)', y=1.02)
plt.tight_layout()
plt.show()


df_corr.to_csv('/Users/mannes/thesis/correlation_evaluation/correlation_table.csv', index=False)
