import json
from pathlib import Path
import tempfile
import unittest
from scripts.build_research_atlas import read_contest


class ContestPayloadTests(unittest.TestCase):
    def payload(self):
        return {'assignments':{mode:{'tid_1':0,'tid_2':1} for mode in ('raw','relative')},
            'dynamics':{mode:{'transition_matrix':[[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]} for mode in ('raw','relative')}}

    def test_rejects_lost_territories_in_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'atlas.json'
            data=self.payload()
            data['dynamics']['raw']['transition_matrix'][0][0]=0
            p.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'conserve'):
                read_contest(p,{'tid_1','tid_2'})

    def test_rejects_unknown_map_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            p,m=Path(directory)/'atlas.json',Path(directory)/'map.json'
            p.write_text(json.dumps(self.payload()),encoding='utf-8')
            m.write_text(json.dumps({'paths':[{'id':'tid_999','d':'M0,0Z'}],'matched':1}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'unknown'):
                read_contest(p,{'tid_1','tid_2'},m)

    def test_rejects_unmatched_memberships(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'atlas.json'
            data=self.payload()
            data['assignments']['relative']['tid_999']=data['assignments']['relative'].pop('tid_2')
            p.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'join'):
                read_contest(p,{'tid_1','tid_2'})
