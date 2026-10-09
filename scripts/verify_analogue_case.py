"""Validate the published analogue case against the exact embedded atlas data."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,math,re
ROOT=Path(__file__).resolve().parents[1]

def verify(refresh_source=False):
    path=ROOT/'docs/index.html';raw=path.read_bytes()
    packed=re.search(rb'\{"encoding":"gzip-base64","data":"([A-Za-z0-9+/=]+)"\}',raw)
    data=json.loads(gzip.decompress(base64.b64decode(packed[1])))
    case_path=ROOT/'reports/round2-cases/abazinsky.json';case=json.loads(case_path.read_text('utf-8'))
    entities={e['id']:e for e in data['entities']};selected=entities[case['selected']['entity_id']]
    distance=lambda e:math.sqrt(sum((a-b)**2 for a,b in zip(selected['features'],e['features'])))
    peers=sorted((e for e in entities.values() if e['id']!=selected['id']),key=lambda e:(distance(e),e['id']))[:15]
    assert [e['id'] for e in peers]==[r['entity_id'] for r in case['neighbors']], 'Nearest-neighbour order differs'
    own=data['contest']['comparability'][selected['id']]
    for row in [case['selected'],*case['neighbors']]:
        e=entities[row['entity_id']];details=data['contest']['comparability'][e['id']]
        assert row['name']==e['name'] and row['region']==e['region']
        assert row['municipality_type']==details['municipal_district_type']
        assert row['profile_number']==e['reference_cluster']+1
        assert row['same_municipality_type']==(details['municipal_district_type']==own['municipal_district_type'])
        for value,expected in [(row['distance'],distance(e)),(row['expense_2023_rubles'],details['expense_2023']),
                               (row['expense_ratio_to_abazinsky'],details['expense_2023']/own['expense_2023'])]:
            assert math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-12)
        for j,name in enumerate(['health','marketplaces','food_service','food','transport']):
            assert row['feature_'+name]==e['features'][j]
            assert row[name+'_share_pct']==e['annual_ratios'][j]
            assert math.isclose(row[name+'_delta_pp'],e['annual_ratios'][j]-selected['annual_ratios'][j],abs_tol=1e-12)
    digest=hashlib.sha256(raw).hexdigest()
    if refresh_source:
        case['source']['sha256']=digest
        case_path.write_text(json.dumps(case,ensure_ascii=False,indent=2)+'\n','utf-8')
    assert case['source']['sha256']==digest,'Case must match the current atlas bytes'
    return {'verified_rows':16,'atlas_sha256':digest,'refitting':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-source',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.refresh_source)))
