import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ───────────────────────────────────────────────────────────────────
# 1)  Load & compute correlations
# ───────────────────────────────────────────────────────────────────
df = pd.read_csv('/complex_task_gen/output/evaluation_results.csv')


def calculate_correlations(model_results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ['perplexity', 'mean_logprob', 'min_logprob', 'sum_logprob', 'std_logprob']
    model_results_df['correct_int'] = model_results_df['correct'].astype(int)

    records = []
    for m in metric_cols:
        if model_results_df[m].isnull().any():
            continue
        pr, pp = pearsonr(model_results_df[m], model_results_df['correct_int'])
        sr, sp = spearmanr(model_results_df[m], model_results_df['correct_int'])
        records.append(dict(metric=m, pearson=pr, pearson_p=pp, spearman=sr, spearman_p=sp))
    return pd.DataFrame(records)


corr_df = calculate_correlations(df).melt(
    id_vars='metric',
    value_vars=['pearson', 'spearman'],
    var_name='correlation_type',
    value_name='value',
)

# ───────────────────────────────────────────────────────────────────
# 2)  Pretty barplot
# ───────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.2)
sns.set_theme(font='serif', rc={'text.usetex': True})

sns.color_palette('Set2')

g = sns.catplot(
    data=corr_df,
    x='metric',
    y='value',
    hue='correlation_type',
    kind='bar',
    height=4,
    aspect=1.4,
    palette='viridis',
)
g.set_axis_labels('', 'Correlation (r / ρ)')
g.despine(left=True)
g.set(ylim=(-1, 1))
g._legend.set_bbox_to_anchor((1, 0.4))  # (x-offset, y-offset) in axes coords
# g._legend.set_frameon(False)                # optional: no box around legend
g._legend.set_title('')  # optional: remove title
g.set(xlabel='', ylabel='Correlation')

# plt.tight_layout()          # make room for the shifted legend
# plt.show()
for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('logprob_correctness_correlations.pdf')
plt.show()

for metric in ['perplexity', 'mean_logprob', 'min_logprob', 'sum_logprob', 'std_logprob']:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxenplot(x='correct_int', y=metric, data=df, palette='viridis', ax=ax)
    sns.swarmplot(x='correct_int', y=metric, data=df, color='k', alpha=0.4, size=2, ax=ax)
    ax.set_xticklabels(['Incorrect (0)', 'Correct (1)'])
    ax.set_title(f'{metric} by correctness')
    plt.tight_layout()
    fig.savefig(f'{metric}_distribution.pdf')
    plt.close(fig)

import matplotlib.pyplot as plt
import seaborn as sns

melted = df.melt(
    id_vars='correct_int',
    value_vars=['perplexity', 'mean_logprob', 'min_logprob', 'sum_logprob', 'std_logprob'],
    var_name='metric',
    value_name='value',
)

g = sns.FacetGrid(melted, row='metric', hue='correct_int', aspect=3, height=1.1, palette='viridis', sharex=False)
g.map(sns.kdeplot, 'value', bw_adjust=0.8, fill=True, clip_on=False, alpha=0.7, linewidth=1)
g.map(sns.kdeplot, 'value', bw_adjust=0.8, color='w', lw=1)  # white outline for ridges
g.fig.subplots_adjust(hspace=-0.8)
g.set_titles(row_template='{row_name}')
g.set(yticks=[])
g.despine(left=True)
g.fig.text(0.9, 0.8, 'Correct = 1\nIncorrect = 0', ha='right')
plt.tight_layout()
plt.show()
plt.savefig('ridgeplot_logprob.pdf')


import numpy as np
from scipy.stats import bootstrap

rows = []
for metric in ['mean_logprob', 'min_logprob', 'std_logprob']:
    for c in (0, 1):
        data = df.loc[df['correct_int'] == c, metric].dropna().to_numpy()
        ci = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=10000, method='basic')
        rows.append(
            dict(
                metric=metric,
                correct=c,
                mean=data.mean(),
                ci_low=ci.confidence_interval.low,
                ci_high=ci.confidence_interval.high,
            ),
        )
ci_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 4))
sns.pointplot(
    data=ci_df,
    x='metric',
    y='mean',
    hue='correct',
    dodge=0.4,
    join=False,
    err_kws={'linewidth': 0},
    palette='viridis',
    ax=ax,
)
# add CIs manually as thin lines
for i, row in ci_df.iterrows():
    ax.vlines(x=i // 2 + (0.2 if row['correct'] == 1 else -0.2), ymin=row['ci_low'], ymax=row['ci_high'], lw=1)
ax.set_ylabel('Mean value (95 % CI)')
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
ax.legend_.set_title('Correct')
plt.tight_layout()
plt.show()
plt.savefig('dumbbell_logprob.pdf')


fig, axes = plt.subplots(1, 5, figsize=(15, 2.8), sharey=True)
for ax, metric in zip(axes, ['perplexity', 'mean_logprob', 'min_logprob', 'sum_logprob', 'std_logprob'], strict=False):
    sns.regplot(
        x=metric,
        y='correct_int',
        data=df,
        ax=ax,
        logistic=True,
        ci=None,
        scatter_kws=dict(s=10, alpha=0.3),
        line_kws=dict(lw=2),
    )
    ax.set_title(metric)
    ax.set_ylabel('P(correct)' if ax is axes[0] else '')
    ax.set_xlabel('')
plt.tight_layout()
plt.show()
plt.savefig('logistic_curves.pdf')

corr_matrix = df[['perplexity', 'mean_logprob', 'min_logprob', 'sum_logprob', 'std_logprob', 'correct_int']].corr()

plt.figure(figsize=(5, 4))
sns.heatmap(
    corr_matrix,
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='vlag',
    linewidth=0.5,
    square=True,
    cbar_kws={'shrink': 0.75},
)
plt.title('Correlation matrix')
plt.tight_layout()
plt.show()
plt.savefig('corr_heatmap.pdf')

g = sns.displot(
    data=df,
    x='mean_logprob',
    col='difficulty_level',
    hue='correct_int',
    kind='kde',
    fill=True,
    facet_kws=dict(sharex=True),
    height=3,
    aspect=0.9,
    palette='viridis',
)
g.set_axis_labels('mean_logprob', 'density')
g.fig.tight_layout()
plt.show()
g.fig.savefig('facet_difficulty_kde.pdf')


g = sns.pairplot(
    df[['perplexity', 'mean_logprob', 'min_logprob', 'sum_logprob', 'std_logprob', 'correct_int']],
    hue='correct_int',
    diag_kind='kde',
    corner=True,
    plot_kws=dict(alpha=0.3, s=15),
    palette='viridis',
)
g.fig.tight_layout()
plt.show()
g.fig.savefig('pairplot_logprob.pdf')
