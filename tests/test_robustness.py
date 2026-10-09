import json
import unittest
from pathlib import Path

import numpy as np

from sbercluster.robustness import match_groups, nesting, one_se, select_k

ROOT = Path(__file__).resolve().parents[1]


def row(k, sw, sdbw, mq=None, boot=0.9, seed=1.0, share=0.1):
    return {"K": k, "SW": sw, "S_Dbw": sdbw, "S_Dbw_paper": sdbw, "MQ": mq, "bootstrap_ari_median": boot,
            "seed_ari_median": seed, "min_share": share}


class RobustnessTests(unittest.TestCase):
    def test_hungarian_matching_relabels_permuted_groups(self):
        reference = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        labels = np.array([2, 2, 3, 0, 1, 1, 0, 0])
        mapping, table = match_groups(reference, labels, 4)
        self.assertEqual(mapping.tolist(), [3, 2, 0, 1])
        self.assertEqual((mapping[labels] == reference).sum(), 7)
        self.assertEqual(table.sum(), 8)
        with self.assertRaisesRegex(ValueError, "Invalid aligned labels"):
            match_groups(reference, labels + 1, 4)

    def test_k_rule_takes_largest_admissible_interior_optimum(self):
        rule = {"min_bootstrap_ari": 0.8, "min_seed_ari": 0.8, "min_share": 0.01, "min_votes": 2,
                "same_index": {"S_Dbw_paper": "S_Dbw"},
                "indices": {"SW": "max", "S_Dbw": "min", "S_Dbw_paper": "min", "MQ": "max"}}
        rows = [row(2, .40, 1.3, .3), row(3, .30, 1.0, .4, boot=.7), row(4, .25, 1.1, .5), row(5, .24, .9, .6),
                row(6, .23, 1.0, .5, share=.005), row(7, .22, .8, .7)]
        chosen = select_k(rows, rule)
        self.assertEqual(chosen["chosen_K"], 5)
        self.assertEqual(chosen["votes"]["3"], ["S_Dbw", "S_Dbw_paper"])
        self.assertEqual(chosen["vote_count"]["3"], 1)
        self.assertEqual(chosen["vote_count"]["5"], 2)
        for r in rows:
            r["MQ"] = None
        self.assertEqual(select_k(rows, rule)["eligible"], [])
        for r in rows:
            r["MQ"] = .5
        self.assertNotIn(3, chosen["admissible"])
        self.assertNotIn(6, chosen["admissible"])
        rows[3]["bootstrap_ari_median"] = .5
        fallback = select_k(rows, rule)
        self.assertEqual(fallback["chosen_K"], 2)
        self.assertIn("maximal SW", fallback["reason"])

    def test_gap_one_se_and_nesting(self):
        self.assertEqual(one_se([1, 2, 3, 4], [0.1, 0.5, 0.55, 0.6], [0.01, 0.01, 0.1, 0.01]), 2)
        self.assertIsNone(one_se([1, 2], [0.1, 0.5], [0.01, 0.01]))
        result = nesting([0, 0, 0, 1, 1, 1], [0, 0, 1, 2, 2, 1])
        self.assertEqual(result["fine_by_coarse"], [[2, 0], [1, 1], [0, 2]])
        self.assertAlmostEqual(result["purity"], 5 / 6)

    def test_config_rule_is_complete(self):
        cfg = json.loads((ROOT / "configs/robustness.json").read_text("utf-8"))
        ks = cfg["k_selection"]
        self.assertEqual(ks["k_grid"], list(range(2, 9)))
        self.assertTrue({"min_bootstrap_ari", "min_seed_ari", "min_share", "min_votes", "indices"} <= ks["rule"].keys())
        self.assertEqual(set(cfg["bases"]), {"annual_2023", "h1_2023", "h2_2023"})


if __name__ == "__main__":
    unittest.main()
