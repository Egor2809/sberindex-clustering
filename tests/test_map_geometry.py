import unittest
from shapely import Polygon, box, count_coordinates, coverage_is_valid

from scripts.build_map import simplify_shared_boundaries


def matched_pair():
    shared = [(1, 0), (1.1, .2), (.9, .4), (1.1, .6), (.9, .8), (1, 1)]
    left = Polygon(
        [(0, 0), *shared, (0, 1), (0, 0)],
        holes=[[(.2, .2), (.4, .2), (.4, .4), (.2, .4), (.2, .2)]],
    )
    right = Polygon([shared[0], (2, 0), (2, 1), shared[-1], *reversed(shared[1:-1]), shared[0]])
    return left, right


class MapGeometryTests(unittest.TestCase):
    def test_simplifies_one_shared_edge_and_preserves_hole(self):
        left, right = matched_pair()
        source = {"left": left, "right": right}
        self.assertTrue(coverage_is_valid(list(source.values())))

        result, audit = simplify_shared_boundaries(source, .2)

        self.assertTrue(coverage_is_valid(list(result.values())))
        self.assertLess(count_coordinates(list(result.values())), count_coordinates(list(source.values())))
        self.assertGreater(result["left"].boundary.intersection(result["right"].boundary).length, .99)
        self.assertEqual(len(result["left"].interiors), 1)
        self.assertTrue(result["left"].interiors[0].equals_exact(left.interiors[0], 0))
        self.assertTrue(audit["outer_boundaries_preserved"])
        self.assertTrue(audit["source_coverage_valid"])
        self.assertEqual(audit["quarantined_source_geometry_count"], 0)

    def test_quarantines_preexisting_overlap_without_new_bad_edges(self):
        left, right = matched_pair()
        source = {
            "left": left,
            "right": right,
            "bad_a": box(10, 0, 11, 1),
            "bad_b": box(10.9, 0, 12, 1),
        }

        result, audit = simplify_shared_boundaries(source, .2)

        self.assertEqual(audit["quarantined_source_geometry_ids"], ["bad_a", "bad_b"])
        self.assertFalse(audit["source_coverage_valid"])
        self.assertTrue(result["bad_a"].equals_exact(source["bad_a"], 0))
        self.assertTrue(result["bad_b"].equals_exact(source["bad_b"], 0))
        self.assertAlmostEqual(
            audit["invalid_edge_length_degrees_after"],
            audit["invalid_edge_length_degrees_before"],
            places=12,
        )
        self.assertGreater(result["left"].boundary.intersection(result["right"].boundary).length, .99)


if __name__ == "__main__":
    unittest.main()
