"""Interpretable fixed-prototype dynamics; no refitting on the examined 2024 values."""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score


def transition_summary(before,after,k=4):
    before,after=np.asarray(before),np.asarray(after)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('Positive integer cluster count required')
    if (before.shape != after.shape or before.ndim != 1 or before.size == 0
            or before.dtype.kind not in 'iu' or after.dtype.kind not in 'iu'
            or before.min()<0 or after.min()<0 or before.max()>=k or after.max()>=k):
        raise ValueError('Invalid aligned labels')
    matrix=np.zeros((k,k),dtype=int)
    np.add.at(matrix,(before,after),1)
    return {'transition_matrix':matrix.tolist(),'changed':int((before!=after).sum()),
            'ari':float(adjusted_rand_score(before,after))}


def drift_analysis(monthly,centers,reference_labels):
    x=np.asarray(monthly,dtype=float)
    centers=np.asarray(centers,dtype=float)
    reference_labels=np.asarray(reference_labels)
    if (x.ndim!=3 or x.shape[0]!=24 or min(x.shape[1:])<1
            or not np.isfinite(x).all()):
        raise ValueError('Expected complete Jan2023-Dec2024 tensor')
    if (centers.ndim!=2 or len(centers)<1 or centers.shape[1]!=x.shape[2]
            or not np.isfinite(centers).all()):
        raise ValueError('Expected finite fixed prototypes in the monthly feature space')
    k=len(centers)
    if (reference_labels.shape!=(x.shape[1],) or reference_labels.dtype.kind not in 'iu'
            or np.any(reference_labels<0) or np.any(reference_labels>=k)):
        raise ValueError('Expected one valid fixed-prototype label per territory')
    def assign(features):
        distance=cdist(features,centers)
        if not np.isfinite(distance).all():
            raise ValueError('Prototype distances overflowed; rescale features and prototypes')
        return distance.argmin(axis=1)
    first,second=x[:12],x[12:]
    y23,y24=np.median(first,axis=0),np.median(second,axis=0)
    raw=assign(y24)
    annual_shift=np.median(y24,axis=0)-np.median(y23,axis=0)
    annual_adjusted=assign(y24-annual_shift)
    monthly_shift=np.median(second,axis=1)-np.median(first,axis=1)
    relative_monthly=second-monthly_shift[:,None,:]
    relative=assign(np.median(relative_monthly,axis=0))
    monthly_labels=np.array([assign(m) for m in x])
    relative_labels=np.array([assign(m) for m in relative_monthly])
    raw_change=raw!=np.asarray(reference_labels)
    relative_change=relative!=np.asarray(reference_labels)
    same_month=[]
    for m in range(12):
        same_month.append({'month':m+1,'raw':transition_summary(monthly_labels[m],monthly_labels[m+12],k),
            'relative':transition_summary(monthly_labels[m],relative_labels[m],k)})
    return {'raw':transition_summary(reference_labels,raw,k),'relative':transition_summary(reference_labels,relative,k),
            'change_overlap':{'both':int((raw_change&relative_change).sum()),
                'raw_only':int((raw_change&~relative_change).sum()),'relative_only':int((~raw_change&relative_change).sum())},
            'annual_median_adjustment':transition_summary(reference_labels,annual_adjusted,k),
            'annual_shift_standardized':annual_shift.tolist(),'same_month_shift_standardized':monthly_shift.tolist(),
            'same_month_transitions':same_month,
            'monthly_counts':[np.bincount(z,minlength=k).tolist() for z in monthly_labels],
            'note':'Relative mode subtracts the national median change separately for each matching calendar month before annual aggregation. It is a retrospective sensitivity analysis, not a causal decomposition.'},raw,relative


def describe_profiles(panel,ids,labels,centers,annual_features,market,registry):
    from .io import CATEGORIES,TOTAL
    names=['Смешанный потребительский профиль','Преобладание повседневных расходов',
           'Городской сервисный профиль','Низкая интенсивность маркетплейсов']
    subtitles=['Промежуточные значения категорий между повседневным и сервисным профилями.',
        'Относительно больше продовольствия, меньше общепита.',
        'Более высокая интенсивность общепита и уровень безналичных расходов.',
        'Низкое отношение расходов на маркетплейсах к итогу и низкая доступность рынков.']
    p=panel[panel.period.str.startswith('2023')].copy()
    ratios=p[CATEGORIES].div(p[TOTAL],axis=0)*100
    ratios['entity_id']=p.entity_id.values
    annual_ratios=ratios.groupby('entity_id')[CATEGORIES].median().reindex(ids)
    totals=p.groupby('entity_id')[TOTAL].median().reindex(ids)
    meta=p.groupby('entity_id').tail(1).set_index('entity_id').reindex(ids)
    ma=market.set_index('territory_id').market_access
    active=registry[(registry.year_from<=2023)&(registry.year_to>2023)].copy()
    if active.territory_id.duplicated().any():
        raise ValueError('Ambiguous2023 registry rows')
    types=active.set_index('territory_id').municipal_district_type
    profiles=[]
    for c in range(4):
        ix=np.where(np.asarray(labels)==c)[0]
        info=meta.iloc[ix]
        market_values=info.territory_id.map(ma).dropna()
        type_values=info.territory_id.map(types)
        nearest=ix[np.argsort(cdist(annual_features[ix],np.asarray(centers)[c:c+1]).ravel(),kind='stable')[:3]]
        profiles.append({'id':c,'name':names[c],'subtitle':subtitles[c],'n':len(ix),
            'median_ratios':annual_ratios.iloc[ix].median().to_list(),'median_total':float(totals.iloc[ix].median()),
            'market_access':float(market_values.median()) if len(market_values) else None,
            'market_access_n':len(market_values),'urban_share':None,
            'municipal_types':dict(Counter(str(x) for x in type_values.dropna())),
            'top_regions':[{'name':str(r),'n':int(n)} for r,n in info.region_name.value_counts().head(5).items()],
            'examples':[{'id':ids[j],'name':str(meta.iloc[j].mo),'region':str(meta.iloc[j].region_name)} for j in nearest]})
    return profiles
