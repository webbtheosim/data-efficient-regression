import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

merged = pd.read_csv('./data/feature_coverage_data.csv', index_col=0)

# Visualize correlation between coverage and performance.
from mpl_toolkits.mplot3d import Axes3D 
plt.rcParams['font.family'] = 'Helvetica Neue'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
fig = plt.figure(figsize=(3.5,3.5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=-45)

random_idx = -1
cluster_idx = -1
adaptive_idx = -1
maximin_idx = -1
al_colors = []
for iter, row in merged.iterrows():
    if 'cluster_margin' in row['Batch']:
        al_colors.append("#C9FFFE")
    elif 'pareto' in row['Batch']:
        al_colors.append("#C9E5FF")
    elif 'hallucinate' in row['Batch']:
        al_colors.append("#C9CAFF")
    elif 'topk' in row['Batch']:
        al_colors.append("#E3C9FF")
    elif row['Strategy'] == 'sf' and row['Sampler'] == 'random':
        random_idx = iter
    elif row['Strategy'] == 'sf' and row['Sampler'] == 'medoids':
        cluster_idx = iter
    elif row['Strategy'] == 'sf_adaptive':
        adaptive_idx = iter
    elif row['Strategy'] == 'sf' and row['Sampler'] == 'maximin':
        maximin_idx = iter

# Plot AL algorithms.
ax.scatter(
    merged[merged['Strategy'] == 'al_adaptive']['coverage-mean'],
    merged[merged['Strategy'] == 'al_adaptive']['feature-mean'],
    merged[merged['Strategy'] == 'al_adaptive']['metric-mean'],
    s=20, color=al_colors, edgecolors='black', linewidth=0.8, zorder=5
)

# Plot SF algorithms.
ax.scatter(
    merged[(merged['Strategy'] == 'sf') & ((merged['Sampler'] == 'maximin') | (merged['Sampler'] == 'max_entropy'))]['coverage-mean'], 
    merged[(merged['Strategy'] == 'sf') & ((merged['Sampler'] == 'maximin') | (merged['Sampler'] == 'max_entropy'))]['feature-mean'],
    merged[(merged['Strategy'] == 'sf') & ((merged['Sampler'] == 'maximin') | (merged['Sampler'] == 'max_entropy'))]['metric-mean'], 
    s=20, color='#FFE3C9', edgecolors='black', linewidth=0.8, zorder=10
)

# Plot random selection.
ax.scatter(
    merged.iloc[random_idx]['coverage-mean'],
    merged.iloc[random_idx]['feature-mean'],
    merged.iloc[random_idx]['metric-mean'],
    marker='^',
    s=20, color='#FFE3C9', edgecolors='black', linewidth=0.8, zorder=10,
)

# Plot cluster selection.
ax.scatter(
    merged.iloc[cluster_idx]['coverage-mean'],
    merged.iloc[cluster_idx]['feature-mean'],
    merged.iloc[cluster_idx]['metric-mean'],
    marker='*',
    s=60, color='#FFE3C9', edgecolors='black', linewidth=0.8, zorder=10,
)
# ax.scatter(
#     merged.iloc[maximin_idx]['coverage-mean'],
#     merged.iloc[maximin_idx]['feature-mean'],
#     merged.iloc[maximin_idx]['metric-mean'],
#     marker='o',
#     s=30, color='#FFFFC9', edgecolors='black', linewidth=0.8, zorder=10,
# )

# Plot adaptive space-filling.
# ax.scatter(
#     merged.iloc[adaptive_idx]['coverage-mean'],
#     merged.iloc[adaptive_idx]['feature-mean'],
#     merged.iloc[adaptive_idx]['metric-mean'],
#     marker='s',
#     s=30, color='#FFE3C9', edgecolors='black', linewidth=0.8, zorder=10,
# )

# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D(
#         [0], [0],
#         marker='o',
#         color='none',
#         label='Cluster-Margin',
#         markerfacecolor='#C9FFFE',
#         markeredgecolor='black',
#         markersize=5,
#         mew=0.8
#     ),
#     Line2D(
#         [0], [0],
#         marker='o',
#         color='none',
#         label='Multiobjective',
#         markerfacecolor='#C9E5FF',
#         markeredgecolor='black',
#         markersize=5,
#         mew=0.8
#     ),
#     Line2D(
#         [0], [0],
#         marker='o',
#         color='none',
#         label='Hallucinate',
#         markerfacecolor='#C9CAFF',
#         markeredgecolor='black',
#         markersize=5,
#         mew=0.8
#     ),
#     Line2D(
#         [0], [0],
#         marker='o',
#         color='none',
#         label='Margin',
#         markerfacecolor='#E3C9FF',
#         markeredgecolor='black',
#         markersize=5,
#         mew=0.8
#     ),
#     Line2D(
#         [0], [0],
#         marker='o',
#         color='none',
#         label='Space-Filling',
#         markerfacecolor='#FFE3C9',
#         markeredgecolor='black',
#         markersize=5,
#         mew=0.8
#     ),
#     Line2D(
#         [0], [0],
#         marker='^',
#         color='none',
#         label='Random',
#         markerfacecolor='#FFE3C9',
#         markeredgecolor='black',
#         markersize=6,
#         mew=0.8
#     ),
#     Line2D(
#         [0], [0],
#         marker='*',
#         color='none',
#         label='Cluster',
#         markerfacecolor='#FFE3C9',
#         markeredgecolor='black',
#         markersize=8,
#         mew=0.8
#     ),

#     Line2D(
#         [0], [0],
#         marker='s',
#         color='none',
#         label='Adapt. Clust.',
#         markerfacecolor='#FFE3C9',
#         markeredgecolor='black',
#         markersize=6,
#         mew=0.8
#     ),
# ]

ax.set_xlim(xmin=0.0, xmax=16.0)
ax.set_ylim(ymin=1.0, ymax=1.05)
ax.set_zlim(zmin=0.74, zmax=0.92)
ax.set_xlabel(r'$\langle\Gamma$/$\Gamma_{\text{min}}\rangle$')
ax.set_ylabel(r'$\langle\Phi$/$\Phi_{\text{min}}\rangle$')
ax.set_zlabel(r'$\langle\rho/\rho_{\text{max}}\rangle$')
ax.set_xticks(ticks=[0.0, 5.0, 10.0, 15.0], labels=[0.0, 5.0, 10.0, 15.0])
ax.set_yticks(ticks=[1.0, 1.02, 1.04], labels=[1.00, 1.02, 1.04])
ax.set_zticks(ticks=[0.75, 0.80, 0.85, 0.90])
ax.tick_params(axis='both', left=True, bottom=True, width=1.2)

# leg = ax.legend(handles=legend_elements, edgecolor='black', handletextpad=0.05, fontsize=8, fancybox=False)
# from scipy.stats import pearsonr, spearmanr
# r = pearsonr(merged['coverage-mean'], merged['metric-mean']).statistic
# rho = spearmanr(merged['coverage-mean'], merged['metric-mean']).statistic
# ax.text(0.03, 0.08, f'r = {r:.3f}', ha='left', transform=ax.transAxes)
# ax.text(0.03, 0.03, rf'$\rho$ = {rho:.3f}', ha='left', transform=ax.transAxes)

plt.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.85, wspace=0, hspace=0)
plt.savefig('./figures/wcss_high_d_3d.pdf', dpi=500)
plt.show()