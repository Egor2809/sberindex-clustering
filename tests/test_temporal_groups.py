import json
import unittest
from pathlib import Path

from sbercluster.temporal_groups import event_owners, flows, rejoin, track_key, track_labels

ROOT = Path(__file__).resolve().parents[1]


def step(events):
    counts = {kind: sum(e["event"] == kind for e in events) for kind in ("continue", "grow", "shrink", "split", "merge", "birth", "death")}
    return {"counts": counts, "events": events}


class TemporalProfilesTests(unittest.TestCase):
    def test_reused_label_gets_new_identity_and_returning_group_is_rejoined(self):
        labels = [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1]]
        sizes = [{"0": 2, "1": 2}, {"0": 4}, {"0": 2, "1": 2}]
        steps = [step([{"event": "merge", "from": [0, 1], "to": [0], "size_before": 4, "size_after": 4}]),
                 step([{"event": "split", "from": [0], "to": [0, 1], "size_before": 4, "size_after": 4}])]
        owners, marks = event_owners(sizes, steps)
        self.assertEqual(owners, [{0: "0", 1: "0", 2: "0"}, {0: "1"}, {2: "1"}])
        self.assertEqual([m["event"] for m in marks], ["merge", "split"])
        tracks = track_labels(labels, owners)
        self.assertEqual(tracks, [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 2, 2]])
        self.assertEqual(rejoin(tracks, 0.5), [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1]])
        self.assertEqual(rejoin(tracks, 1.01), tracks)
        self.assertEqual(flows(rejoin(tracks, 0.5)), [{(0, 0): 2, (1, 0): 2}, {(0, 0): 2, (0, 1): 2}])
        self.assertEqual(track_key(2, owners, ["2023-01-01", "2023-02-01", "2023-03-01"]), "2023-03:1")
        with self.assertRaisesRegex(ValueError, "no lifecycle track"):
            track_labels([[0, 5, 1, 1], *labels[1:]], owners)

    def test_rejoin_skips_groups_that_overlap_in_time(self):
        tracks = [[0, 0, 1, 1], [0, 1, 2, 2]]
        self.assertEqual(rejoin(tracks, 0.1), tracks)

    def test_published_summary_names_every_stable_group(self):
        summary = json.loads((ROOT / "reports/temporal-groups/summary.json").read_text("utf-8"))
        self.assertEqual(summary["stable_groups"], len(summary["groups"]))
        self.assertTrue(all(g["name"] and g["months"] >= summary["min_months"] for g in summary["groups"]))
        self.assertTrue(0 <= summary["ari_reference"]["median"] < 1)
        self.assertEqual(summary["multi_group"], round(summary["multi_group_share"] * summary["n"]))
        self.assertTrue(all(sum(g["types"].values()) > 0.999 for g in summary["groups"]))


if __name__ == "__main__":
    unittest.main()
