#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import seaborn as sb
import pandas as pd
#sns.set(color_codes=True)

from scipy import stats
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 26})
from itertools import cycle
colors =  ["red","black","#2ecc71", "#2e0071",  "#2efdaa", "#200daa","#2ffd00"]


# In[98]:


def five_perc_diff(df):
    
    top_hit_rate = df.loc[0, 'hit_rate']
    c_hit_rate = df.loc[df['algo'] == 'cacheus', 'hit_rate'].iloc[0]
    
    diff = 0
    is_top = False
    
    if df.loc[0, 'algo'] == 'cacheus':
        is_top = True
    elif (abs(top_hit_rate - c_hit_rate)) / top_hit_rate <= 0.05:
        is_top = True
    
    if is_top:
        for _, row in df.iterrows():
            if (top_hit_rate - row['hit_rate']) / top_hit_rate > 0.05:
                return c_hit_rate - row['hit_rate']
    else:
        return 0

    return 0

def five_perc_diff_neg(df):
    
    top_hit_rate = df.loc[0, 'hit_rate']
    c_hit_rate = df.loc[df['algo'] == 'cacheus', 'hit_rate'].iloc[0]
    
    diff = 0
    is_top = True
    
    
    if df.loc[0, 'algo'] == 'cacheus':
        is_top = True
        return 0
    
    elif (top_hit_rate - c_hit_rate) / top_hit_rate > 0.05:
        return top_hit_rate - c_hit_rate
    
    return 0

def filter_algs(df):
    filt = (df['algo'] == 'cacheus') | (df['algo'] == 'arcalecar') | (df['algo'] == 'lirsalecar')
    new_algs = df[filt]
    ret_algs = None
    
    top_hit_rate = df.loc[0, 'hit_rate']
    best_algs = df[~filt]
    best = None
    for _, row in best_algs.iterrows():
        if (top_hit_rate - row['hit_rate']) / top_hit_rate > 0.05:
            best = row.copy()
            break
    
    if best is None:
        best = df[~filt].iloc[0]
    #print(best['algo'])
    best['algo'] = 'top'
    
    return new_algs.append(best).sort_values(by=['hit_rate'], ascending=False).reset_index(drop=True)


# In[100]:

class HitRatePlugin:
 def input(self, inputfile):
  self.nums_df = pd.read_csv(inputfile, header=None, index_col=None)
 def run(self):
     pass
 def output(self, outputfile):
  self.nums_df.columns = ['traces', 'trace_name', 'algo', 'hits', 'misses', 'writes', 'filters', 
                      'size','cache_size', 'requestcounter', 'hit_rate', 'time', 'dataset']

  useful_cols = ['dataset', 'cache_size', 'algo','traces', 'hit_rate']
  df = self.nums_df[useful_cols]
  df = df.astype({'cache_size' : float, 'hit_rate' : float})


  df_grouped = df.groupby(['dataset', 'cache_size'])

  ret_df = pd.DataFrame(columns=useful_cols)
  ret_df_list = []
  for name, group in df_grouped:
    
    ggrouped_df = group.groupby(['traces'])
    max_diff = 0
    max_group = None
    plan_b_in_name = ""
    print(name)
    for in_name, in_group in ggrouped_df:
        plan_b_in_name = in_name
        in_group = in_group.sort_values(by=['hit_rate'], ascending=False, inplace=False).reset_index(drop=True)
        
        diff_with_second = five_perc_diff_neg(in_group)
    
#         print(in_group)
        
        if diff_with_second > max_diff:
            max_diff = diff_with_second
            max_group = in_group.copy()
            
    # print('diff: {}'.format(max_diff))
    # corner cases when there is not top or worst
    if max_diff == 0:
        min_diff = 100
        for in_name, in_group in ggrouped_df:
            in_group = in_group.sort_values(by=['hit_rate'], ascending=False, inplace=False).reset_index(drop=True)
            top_hit_rate = in_group.loc[0, 'hit_rate']
            c_hit_rate = in_group.loc[in_group['algo'] == 'cacheus', 'hit_rate'].iloc[0]
            diff = top_hit_rate - c_hit_rate
            if diff > max_diff: # reverse this. replace max_diff with min_diff
                max_diff = diff
                max_group = in_group.copy()
    
    #if max_diff == 0:
    #    print('plan b')
    #    max_group = ggrouped_df.get_group(plan_b_in_name).assign(traces=None, hit_rate=0).reset_index(drop=True)

    print("\nmax_trace")
    print(filter_algs(max_group))
    ret_df_list.append(filter_algs(max_group))
    
  ret_df = ret_df.append(ret_df_list).reset_index(drop=True)
  #print(ret_df)
  algos = df["algo"].unique()

  #print(df_grouped)
  #print(algos)


  # In[101]:


  # plt.style.use('seaborn-darkgrid')
  # plt.rc('grid', linestyle="-", color='white')
  # # sns.set(style='ticks') 
  sb.set(font_scale=1.5)

  # plt.rcParams['hatch.linewidth'] = 3

  # ret_df['algo'] = pd.Categorical(ret_df['algo'], ["arcalecar", "lirsalecar", "cacheus", "top"])

  ret_df['algo'] = ret_df['algo'].replace({'arcalecar': 'C1', 'lirsalecar': 'C2', 'cacheus': 'C3', 'top': 'Non-Cacheus Best'})

  ret_df['algo'] = pd.Categorical(ret_df['algo'], ["C1", "C2", "C3", "Non-Cacheus Best"])


  g = sb.catplot(x='cache_size', y='hit_rate', col='dataset', hue='algo', data=ret_df, kind='bar', 
                     col_wrap=5, height=5, aspect=1, ci=None, legend=False, palette='muted')
  # sns.despine(ax=ax, left=True)
  axes = g.axes.flatten()
  axes[0].set_ylabel("Hit-rate (%)")

  labels = ["0.05", "0.1", "0.5", "1", "5", "10"]
  axes[0].set_xticklabels(labels)
  axes[0].set_xlabel("")
  axes[0].set_title("CloudCache")
  axes[1].set_xticklabels(labels)
  axes[1].set_xlabel("")
  axes[1].set_title("CloudPhysics")
  axes[2].set_xticklabels(labels)
  axes[2].set_xlabel("")
  axes[2].set_title("CloudVPS")
  axes[3].set_xticklabels(labels)
  axes[3].set_xlabel("")
  axes[3].set_title("FIU")
  axes[4].set_xticklabels(labels)
  axes[4].set_xlabel("")
  axes[4].set_title("MSR")

  num_locations = len(ret_df.cache_size.unique())
  hatches = cycle(['/', 'xx', '\\', ' '])
  for i, bar in enumerate(axes[0].patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    bar.set_hatch(hatch)
    bar.set_edgecolor('k')
    bar.set_linewidth(2)    
  for i, bar in enumerate(axes[1].patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    bar.set_hatch(hatch)
    bar.set_edgecolor('k')
    bar.set_linewidth(2)
  for i, bar in enumerate(axes[2].patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    bar.set_hatch(hatch)
    bar.set_edgecolor('k')
    bar.set_linewidth(2)
  for i, bar in enumerate(axes[3].patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    bar.set_hatch(hatch)
    bar.set_edgecolor('k')
    bar.set_linewidth(2)
  for i, bar in enumerate(axes[4].patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    bar.set_hatch(hatch)
    bar.set_edgecolor('k')
    bar.set_linewidth(2)
    
  #axes[1].set_title("External")

  # axes[0].legend(loc='upper right', bbox_to_anchor=(-5, -5), ncol=1, fancybox=True, shadow=True)
  axes[0].legend(loc='upper left', fancybox=True)


  # fig.show()

  #plt.savefig('best_case_plot.png', format="png", dpi=300)


  # In[ ]:




