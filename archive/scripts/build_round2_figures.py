"""Plots from completed comparisons; no model fitting or parameter selection."""
from pathlib import Path
import json,os
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/round2-2026-09-24';OUT=ROOT/'docs/images'
def read(stage,name):return json.loads((R/('round2-20260924-'+stage)/name).read_text('utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf6','axes.facecolor':'#fafaf6'})
rows={r['id']:r for r in read('region_control','metrics.json')}
ext={r['id']:r for r in read('region_control','external_comparison.json')['results'] if r['subset']=='all'}
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
keys=['kmeans_k4','joint_same_region_k4','joint_road_k4'];names=['Только расходы','Расходы + регион','Расходы + дороги'];colors=['#285f8f','#ac7f39','#076a58']
for i,key in enumerate(keys):
    v=rows[key]['on_real_road']['graph_cut_fraction'];axes[0].bar(i,v,color=colors[i],width=.6);axes[0].text(i,v+.009,f'{v:.3f}',ha='center')
    v=ext[key]['partial_R2'];lo,hi=ext[key]['ci95'];axes[1].bar(i,v,color=colors[i],width=.6);axes[1].vlines(i,lo,hi,color='#35443c');axes[1].plot([i-.05,i+.05],[lo,lo],color='#35443c');axes[1].plot([i-.05,i+.05],[hi,hi],color='#35443c')
for ax in axes:ax.set_xticks(range(3),names);ax.grid(axis='y',alpha=.15)
axes[0].set(title='K=4: связность на настоящих дорогах',ylabel='Межгрупповая доля дорожного веса (меньше)')
axes[1].set(title='Доступность рынков после контролей',ylabel='Partial R² (описательная ассоциация)')
fig.suptitle('Проверка альтернативного объяснения: достаточно ли региона?',fontsize=15)
fig.text(.5,-.10,'Контроли: регион × тип МО + уровень расходов. 95% bootstrap по регионам, фиксированные метки.\nДорожные связи и доступность рынков частично отражают одну инфраструктуру.',ha='center',fontsize=9)
fig.savefig(OUT/'round2-region-control.png',dpi=160,bbox_inches='tight');plt.close(fig)

rows=read('dmon','metrics.json');fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,k in zip(axes,[2,4]):
    for i,(prefix,title,col) in enumerate([('kmeans_','KMeans','#285f8f'),('joint_road_','Расходы + дороги','#076a58'),('dmon_','DMoN · 3 seed','#7b426f')]):
        selected=[r for r in rows if r['id'].startswith(prefix) and (r['id']==prefix+f'k{k}' or f'k{k}_seed' in r['id'])]
        vals=[r['SW'] for r in selected if r['SW'] is not None]
        ax.scatter(np.full(len(vals),i)+np.linspace(-.08,.08,len(vals)),vals,color=col,s=65,zorder=3)
        if vals:ax.hlines(np.mean(vals),i-.22,i+.22,color=col,lw=2)
    ax.set_xticks(range(3),['KMeans','Расходы + дороги','DMoN · 3 seed']);ax.set(title=f'{k} запрошенных группы',ylabel='Силуэт на общих признаках (больше)');ax.axhline(0,color='#999',lw=.8);ax.grid(alpha=.15)
fig.suptitle('DMoN по формуле статьи: тот же набор признаков и дорожный граф',fontsize=14)
fig.text(.5,-.07,'1 000 эпох; checkpoint выбран по обучающему loss. Внешние показатели не участвуют в выборе.',ha='center',fontsize=9)
fig.savefig(OUT/'round2-dmon.png',dpi=160,bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,k in zip(axes,[2,4]):
    for seed,col in zip([1729,2718,3141],['#285f8f','#076a58','#7b426f']):
        trace=read('dmon',f'dmon_k{k}_seed{seed}/trace.json')
        ax.plot([r['epoch'] for r in trace],[r['objective'] for r in trace],label=f'seed {seed}',color=col,lw=1.5)
    ax.set(title=f'K={k}',xlabel='Эпоха',ylabel='Обучающая целевая функция (меньше)');ax.grid(alpha=.15);ax.legend()
fig.suptitle('Бюджет исчерпан, сходимость не подтверждена',fontsize=15)
fig.text(.5,-.07,'Во всех шести запусках лучший loss достигнут на эпохе 1 000. Это сравнение при фиксированном бюджете.',ha='center',fontsize=9)
fig.savefig(OUT/'round2-dmon-traces.png',dpi=160,bbox_inches='tight');plt.close(fig)

rows=read('temporal_grid','temporal_summary.json');stability=read('temporal_grid','temporal_stability.json');omegas=sorted(set(r['omega'] for r in rows))
fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
for ax,field,title in [(axes[0],'mean_churn','Доля смен между месяцами'),(axes[1],'mean_SW','Силуэт в исходных признаках')]:
    values=np.array([[r[field] for r in rows if r['omega']==o] for o in omegas]);ax.plot(omegas,values.mean(axis=1),'o-',color='#076a58');ax.fill_between(omegas,values.min(axis=1),values.max(axis=1),color='#076a58',alpha=.17);ax.set(title=title,xlabel='Вес временной связи ω');ax.grid(alpha=.15)
axes[2].plot([r['omega'] for r in stability],[r['pairwise_seed_monthly_ARI_mean'] for r in stability],'o-',color='#7b426f');axes[2].set(title='Согласие разбиений между seed',xlabel='Вес временной связи ω',ylabel='Средний ARI');axes[2].grid(alpha=.15)
fig.suptitle('24 месяца × 5 весов связи × 3 seed',fontsize=15)
fig.text(.5,-.04,'Полоса — разброс между тремя seed. Сглаживание поощряется методом и само по себе не доказывает правильность переходов.',ha='center',fontsize=9)
fig.savefig(OUT/'round2-temporal.png',dpi=160,bbox_inches='tight');plt.close(fig)
print('Created four round-two figures')
