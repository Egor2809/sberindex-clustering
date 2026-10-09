"""Draw figures exclusively from completed published research artifacts."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--research-dir',type=Path,default=ROOT/'reports/review-2026-09-24')
parser.add_argument('--external-results',type=Path,default=ROOT/'reports/external-v4/results.json')
parser.add_argument('--output-dir',type=Path,default=ROOT/'docs/images')
args=parser.parse_args()
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
R=args.research_dir
rows=json.loads((R/'review-20260924-joint/metrics.json').read_text('utf-8'))
by={r['id']:r for r in rows}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf6','axes.facecolor':'#fafaf6','axes.titleweight':'bold'})
fig,ax=plt.subplots(figsize=(9,5.4),layout='constrained')
for name,label,color,offset in [('joint_alpha0_k4','Только расходы · α=0','#285f8f',(-3,13)),('joint_road_k4','Расходы + дороги · α=0,25','#076a58',(12,-16)),('joint_road_sensitivity_k4','Сильный вес дорог · α=1','#b26b20',(-175,13))]:
 f=by[name]['fit_info'];x=f['attribute_SSE_over_TSS'];y=f['graph_cut_fraction'];ax.scatter(x,y,s=105,color=color,zorder=3);ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=10)
for i in [1730,1731,1732]:
 f=by['joint_region_shuffled_'+str(i)]['fit_info']['objective_on_real_road'];ax.scatter(f['attribute_SSE_over_TSS'],f['graph_cut_fraction'],marker='x',s=75,color='#7b426f',label='Перестановки внутри региона' if i==1730 else None)
ax.set(xlabel='Разброс расходов внутри групп / общий разброс → хуже',ylabel='Доля веса дорожных рёбер между группами → хуже',title='Четыре группы: что меняется при добавлении дорог')
ax.legend(loc='upper right',frameon=False);ax.grid(alpha=.15);fig.savefig(OUT/'joint-tradeoff.png',dpi=165);plt.close(fig)

base=json.loads((ROOT/'reports/contest-v3/descriptive.json').read_text('utf-8'))
counts=[base['profiles'][3]['n'],sum(row[3] for row in base['dynamics']['raw']['transition_matrix']),sum(row[3] for row in base['dynamics']['relative']['transition_matrix'])]
fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
axes[0].bar(['2023','2024\nисходная шкала','2024\nотносительный сдвиг'],counts,color=['#7b426f','#b18baa','#076a58'],width=.62)
for i,n in enumerate(counts):axes[0].text(i,n+3,str(n),ha='center',fontsize=15)
axes[0].set(ylabel='Число территорий',ylim=(0,155),title='Группа с низкой интенсивностью маркетплейсов')
ratios=[base['dynamics']['national_ratios_2023'][1],base['dynamics']['national_ratios_2024'][1]]
axes[1].bar(['2023','2024'],ratios,color=['#285f8f','#076a58'],width=.5)
for i,n in enumerate(ratios):axes[1].text(i,n+.35,f'{n:.2f}%',ha='center',fontsize=15)
axes[1].set(ylabel='Медианное отношение категории к итогу, %',ylim=(0,19),title='Рост маркетплейсов по всей панели')
fig.savefig(OUT/'marketplace-shift.png',dpi=165);plt.close(fig)

r=json.loads(args.external_results.read_text('utf-8'))
cases=['none','region','region_type','region_type_level'];names=['Без контроля','Регион','Регион × тип МО','Регион × тип МО\n+ уровень расходов']
selected=[next(x for x in r['effects'] if x['subset']=='all' and x['outcome']=='market_access' and x['controls']==c) for c in cases]
y=np.array([x['partial_r_squared'] for x in selected]);lo=np.array([x['region_bootstrap_percentile_95'][0] for x in selected]);hi=np.array([x['region_bootstrap_percentile_95'][1] for x in selected])
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');ax.bar(names,y,color=['#285f8f','#668eae','#519c89','#076a58'],width=.6);ax.errorbar(range(4),y,yerr=[y-lo,hi-y],fmt='none',color='#34473f',capsize=5)
for i,v in enumerate(y):ax.text(i+.21,v+.016,f'{v:.3f}',ha='left')
ax.set(title='Связь опорных групп с доступностью рынков',ylabel='Partial R² относительно остаточной вариации',ylim=(0,.7));ax.text(.98,.96,'n=2 004; 500 региональных bootstrap\n95% интервалы условны на фиксированные метки',transform=ax.transAxes,ha='right',va='top',fontsize=9);fig.savefig(OUT/'external-controls.png',dpi=165);plt.close(fig)
print('Created three result figures')
