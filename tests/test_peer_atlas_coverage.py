import unittest
import argparse
from pathlib import Path
from scripts.build_research_atlas import build
from scripts.build_research_atlas import validate_neighbor_coverage


class PeerAtlasCoverage(unittest.TestCase):
    def fixture(self):
        ids=[str(i) for i in range(16)]
        result={i:[{'id':j,'rank':rank+1} for rank,j in enumerate(k for k in ids if k!=i)] for i in ids}
        result['excluded']=[]
        return result,{'n_common':16,'n_excluded':1}

    def test_validation_cannot_display_unrelated_pilot_metrics(self):
        args=argparse.Namespace(reference_run=None,validation_run=Path('irrelevant'))
        with self.assertRaisesRegex(ValueError,'requires --reference-run'):
            build(args)

    def test_explicitly_excluded_territory_is_allowed(self):
        neighbors,report=self.fixture()
        validate_neighbor_coverage(neighbors,report)

    def test_missing_source_cannot_silently_change_cohort(self):
        neighbors,report=self.fixture();neighbors['0']=[]
        with self.assertRaisesRegex(ValueError,'coverage'):
            validate_neighbor_coverage(neighbors,report)

    def test_repeated_peer_is_not_fifteen_distinct_analogs(self):
        neighbors,report=self.fixture();neighbors['0'][1]['id']=neighbors['0'][0]['id']
        with self.assertRaisesRegex(ValueError,'distinct'):
            validate_neighbor_coverage(neighbors,report)


if __name__=='__main__':unittest.main()
