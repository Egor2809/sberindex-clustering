"""Build comparisons from completed DMoN evidence; no fitting or selection."""
from pathlib import Path
import gzip,json,os
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/dmon-plateau-2026-09-25'
NEW=OUT/'dmon-plateau-20260925'
OLD=ROOT/'reports/round2-2026-09-24/round2-20260924-dmon'

def read(path):
    raw=path.read_bytes();return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)

def main():
    assert read(NEW/'status.json')['status']=='completed'
    old={r['id']:r for r in read(OLD/'metrics.json')};new={r['id']:r for r in read(NEW/'metrics.json')}
    external={r['id']:r for r in read(NEW/'external_comparison.json')['results'] if r['subset']=='all'}
    rows=[];prefix=[]
    for k in [2,4]:
        for seed in [1729,2718,3141]:
            name=f'dmon_k{k}_seed{seed}';info=read(NEW/name/'info.json');trace=read(NEW/name/'trace.json.gz')
            before=np.load(OLD/name/'labels.npy',allow_pickle=False);replay=np.load(NEW/name/'epoch1000_labels.npy',allow_pickle=False)
            prefix.append({'id':name,'epoch':1000,'exact_labels':bool(np.array_equal(before,replay)),
                           'different_labels':int(np.count_nonzero(before!=replay)),
                           'ARI':float(adjusted_rand_score(before,replay)),
                           'objective_difference':trace[1000]['objective']-old[name]['fit']['trace'][1000]['objective']})
            rows.append({'id':name,'requested_k':k,'seed':seed,'epochs':info['epochs'],'best_epoch':info['best_epoch'],
                         'stop_reason':info['stop_reason'],'plateau_reached':info['plateau_reached'],
                         'SW_1000':old[name]['SW'],'SW_plateau':new[name]['SW'],
                         'loss_1000':old[name]['fit']['best_objective'],'loss_plateau':info['best_objective'],
                         'road_cut_1000':old[name]['on_real_road']['graph_cut_fraction'],
                         'road_cut_plateau':new[name]['on_real_road']['graph_cut_fraction'],
                         'partial_R2':external[name]['partial_R2'],'delta_to_kmeans_ci95':external[name]['delta_ci95'],
                         'occupied_k':info['occupied_clusters'],'training_seconds':info['training_seconds']})
    for name,data in [('comparison.json',rows),('prefix_comparison.json',prefix)]:
        (OUT/name).write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf6','axes.facecolor':'#fafaf6'})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,k in enumerate([2,4]):
        ax=axes[0,col]
        for seed,color in zip([1729,2718,3141],['#285f8f','#076a58','#7b426f']):
            name=f'dmon_k{k}_seed{seed}';trace=read(NEW/name/'trace.json.gz')
            ax.plot([r['epoch'] for r in trace],[r['objective'] for r in trace],color=color,label=f'seed {seed}',lw=1)
        ax.axvline(1000,color='#888',ls='--',lw=1,label='Прежний бюджет');ax.set(title=f'K={k}: полный обучающий loss',xlabel='Эпоха',ylabel='Меньше — лучше');ax.legend(fontsize=8);ax.grid(alpha=.15)
        ax=axes[1,col];selected=[r for r in rows if r['requested_k']==k]
        for row,color in zip(selected,['#285f8f','#076a58','#7b426f']):
            ax.plot([0,1],[row['SW_1000'],row['SW_plateau']],'o-',color=color,label=f"seed {row['seed']}")
        ax.axhline(old[f'kmeans_k{k}']['SW'],color='#35443c',ls='--',label='KMeans');ax.set_xticks([0,1],['1 000 эпох','Остановка по правилу плато']);ax.set(title=f'K={k}: силуэт на общих признаках',ylabel='Больше — лучше',ylim=(0,.4));ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('DMoN: что изменилось после продолжения обучения',fontsize=16)
    fig.savefig(ROOT/'docs/images/dmon-plateau.png',dpi=155,bbox_inches='tight');plt.close(fig)
    print(json.dumps({'models':len(rows),'plateau_reached':sum(r['plateau_reached'] for r in rows),'epochs':[r['epochs'] for r in rows],'all_prefix_labels_exact':all(r['exact_labels'] for r in prefix)},ensure_ascii=False))

if __name__=='__main__':main()
