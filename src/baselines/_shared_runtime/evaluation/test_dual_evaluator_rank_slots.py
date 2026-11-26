import unittest

from rdkit import RDLogger

import evaluate_nag2g_dual as nag2g
import evaluate_ranked_precursors_dual as generic
import evaluate_transgraphedit_dual as transgraphedit


class CanonicalRankSlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RDLogger.DisableLog("rdApp.*")

    def test_generic_invalid_and_duplicate_predictions_keep_raw_slots(self):
        ranked, invalid, duplicates = generic.ranked_canonical_predictions(
            ["invalid", "CC", "CC", "O"]
        )
        self.assertEqual(ranked, [(2, "CC"), (4, "O")])
        self.assertEqual(invalid, 1)
        self.assertEqual(duplicates, 1)

    def test_transgraphedit_hit_uses_original_beam_rank(self):
        ranked, invalid, duplicates = transgraphedit.ranked_canonical_predictions(
            [
                {"smiles": "invalid"},
                {"smiles": "CC"},
                {"smiles": "CC"},
                {"smiles": "O"},
            ]
        )
        self.assertEqual(transgraphedit.hit_rank(ranked, "O"), 4)
        self.assertEqual(invalid, 1)
        self.assertEqual(duplicates, 1)

    def test_nag2g_hit_uses_original_beam_rank(self):
        ranked, _, _ = nag2g.ranked_canonical_predictions(
            ["invalid", "CC", "CC", "O"]
        )
        hits = nag2g.Counter()
        self.assertEqual(nag2g.add_rank_hits(hits, ranked, "O"), 4)
        self.assertEqual(hits, nag2g.Counter({5: 1, 10: 1}))

    def test_legacy_set_is_order_and_map_independent(self):
        left = generic.canonicalize_precursor_set("[CH3:1][OH:2].CC.CC")
        right = generic.canonicalize_precursor_set("CC.CO")
        self.assertEqual(left, right)

    def test_stereochemistry_is_preserved(self):
        clockwise = generic.canonicalize_precursor_set("F[C@H](Cl)Br")
        counterclockwise = generic.canonicalize_precursor_set("F[C@@H](Cl)Br")
        self.assertNotEqual(clockwise, counterclockwise)


if __name__ == "__main__":
    unittest.main()
