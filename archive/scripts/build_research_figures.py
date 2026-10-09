"""Render publication figures from exported results; this script performs no fitting."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
COLORS = {'kmeans':'#167d8d','ward':'#62758d','gmm_full':'#9a7b55',
          'spectral_global':'#ab5e65','spectral_local':'#6579bc',
          'leiden_global':'#9c7bb2','leiden_local':'#65a17d'}
SHORT = {'kmeans_k4':'KMeans · 4','kmeans_k6':'KMeans · 6',
         'spectral_local_k4':'Spectral local · 4','leiden_local_r0p4':'Leiden local · 7'}
PEERS = {'profile_15':'Ближайшие по профилю', 'temporal_consensus_15':'Повторяющиеся соседи',
         'cluster_profile_15':'Ближайшие внутри группы', 'region_all':'Свой регион',
         'level_15':'Похожий уровень расходов', 'geographic_15':'Географические соседи',
         'random_15':'Случайные территории'}


def render(report, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold', 'savefig.facecolor':'#f8fafc',
                         'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc'})
    def save(fig, name, note):
        fig.text(.04,.025,note,fontsize=9,color='#526078',va='bottom')
        fig.savefig(output/(name+'.png'),dpi=160,bbox_inches='tight')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
    rows=json.loads((report/'screen_metrics.json').read_text('utf-8'))
    annual=[r for r in rows if r['representation']=='annual_2023' and r['status']=='completed']
    fig,axes=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1.2,1]})
    for method in COLORS:
        data=sorted([r for r in annual if r['spec']['method']==method],key=lambda r:r['k'])
        axes[0].plot([r['k'] for r in data],[r['SW'] for r in data],'-o',color=COLORS[method],label=method.replace('_',' '),ms=5)
    suspect=[r for r in annual if r['min_cluster_size']<20]
    axes[0].scatter([r['k'] for r in suspect],[r['SW'] for r in suspect],s=130,facecolors='none',edgecolors='#d74343',linewidths=1.5)
    axes[0].set(xlabel='Полученное число групп',ylabel='Силуэт · больше = лучше',title='Высокая метрика может скрывать малые группы')
    axes[0].legend(fontsize=8,ncol=2,loc='upper right');axes[0].grid(alpha=.15)
    candidates=['kmeans_k4','kmeans_k6','kmeans_k8','spectral_local_k4']
    positions=np.arange(len(candidates));width=.22
    for j,period in enumerate(['2023-01-01','2023-06-01','2023-12-01']):
        vals=[next(r['SW'] for r in rows if r['candidate']==c and r['representation']==period) for c in candidates]
        axes[1].bar(positions+(j-1)*width,vals,width,label={'2023-01-01':'Январь','2023-06-01':'Июнь','2023-12-01':'Декабрь'}[period],color=['#124e66','#22899b','#79beb6'][j])
    axes[1].set_xticks(positions,[SHORT.get(c,'KMeans · 8').replace(' · ','\n') for c in candidates])
    axes[1].set(ylabel='Силуэт',title='Проверка на отдельных месяцах 2023 года',ylim=(0,.3));axes[1].legend(fontsize=9);axes[1].grid(axis='y',alpha=.15)
    fig.suptitle('Сравнение 104 разбиений: выбор по данным 2023 года',fontsize=18,x=.04,ha='left')
    fig.subplots_adjust(bottom=.18,top=.84,wspace=.3)
    save(fig,'model-comparison','Красное кольцо: в разбиении есть группа меньше 20 территорий. Все методы сравниваются в одном пространстве признаков.')

    stability=json.loads((report/'stability.json').read_text('utf-8'))
    fig,axes=plt.subplots(1,3,figsize=(13,5.6),sharey=True)
    for ax,kind,title in zip(axes,['seed','month_block','drop_feature'],['Смена seed · 2 повтора','Блоки месяцев · 6 повторов','Убрать категорию · 5 вариантов']):
        for i,c in enumerate(SHORT):
            values=[r['ARI'] for r in stability if r['candidate']==c and r['kind']==kind]
            ax.scatter(np.arange(len(values))*.035+i-.08,values,s=35,color=COLORS[next(r['spec']['method'] for r in annual if r['candidate']==c)],alpha=.8)
            ax.plot([i-.2,i+.2],[np.median(values)]*2,color='#1d2c41',lw=2)
        ax.set_xticks(range(4),[SHORT[c].replace(' · ','\n') for c in SHORT],rotation=25,ha='right',fontsize=9)
        ax.set(title=title,ylim=(0,1.05));ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('ARI · 1 = одинаковые группы')
    fig.suptitle('Устойчивость: что сохраняется, а что зависит от признаков',fontsize=18,x=.04,ha='left')
    fig.subplots_adjust(bottom=.25,top=.8,wspace=.12)
    save(fig,'stability','Точки — реальные возмущения, чёрточка — медиана. Удаление категории меняет смысл сравнения; это проверка чувствительности.')

    validation=report/'validation'
    if not (validation/'peer_validation.json').exists():return
    peers=json.loads((validation/'peer_validation.json').read_text('utf-8'))
    data=peers['results'];fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,key,title in zip(axes,['profile_euclidean_discrepancy','growth_absolute_discrepancy_pp'],['Расхождение профилей в 2024 году','Расхождение роста расходов · п.п.']):
        ordered=sorted(data,key=lambda r:r[key])
        bars=ax.barh([PEERS[r['method']] for r in ordered],[r[key] for r in ordered],color=['#167d8d' if r['method'] in ('profile_15','temporal_consensus_15') else '#aab8c7' for r in ordered])
        ax.invert_yaxis();ax.set_title(title,fontsize=13);ax.set_xlabel('Меньше = лучше');ax.grid(axis='x',alpha=.15)
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=9);ax.set_xlim(0,max(r[key] for r in ordered)*1.18)
    fig.suptitle('Отложенная проверка: с кем разумнее сравнивать территорию',fontsize=17,x=.04,ha='left')
    fig.subplots_adjust(left=.19,right=.97,bottom=.2,top=.82,wspace=.8)
    save(fig,'peer-validation',f"Соседи выбраны по 2023 году; сравниваются наблюдённые результаты 2024 года. Общая выборка: {peers['n_common']} территорий. Это не прогноз.")

    profiles=pd.read_csv(validation/'cluster_profiles.csv')
    categories=['Здоровье','Маркетплейсы','Общественное питание','Продовольствие','Транспорт']
    pivot=profiles.pivot(index='cluster',columns='category',values='median_standardized_log_ratio')[categories]
    sizes=profiles.groupby('cluster')['n'].first()
    fig,ax=plt.subplots(figsize=(11,5.2));lim=float(np.abs(pivot.to_numpy()).max())
    im=ax.imshow(pivot,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(5),['Здоровье','Маркетплейсы','Общепит','Продовольствие','Транспорт'])
    ax.set_yticks(range(len(pivot)),[f'Профиль {c+1} · {sizes.loc[c]} МО' for c in pivot.index])
    for i in range(len(pivot)):
        for j in range(5):
            value=pivot.iloc[i,j];ax.text(j,i,f'{value:+.2f}',ha='center',va='center',color='white' if abs(value)>.65*lim else '#15253a')
    fig.colorbar(im,ax=ax,shrink=.8,label='Стандартизованное log(категория / итог)')
    ax.set_title('Четыре профиля потребления: различия по пяти категориям',pad=20)
    fig.subplots_adjust(bottom=.2,left=.22,top=.82)
    save(fig,'cluster-profiles','Медианы годовых профилей 2023 года. Ноль — общая опорная медиана; это относительные интенсивности, не доли бюджета.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',default='reports/experiments/2026-09-23-v2')
    p.add_argument('--output',default='docs/images')
    a=p.parse_args();render(ROOT/a.report,ROOT/a.output)
