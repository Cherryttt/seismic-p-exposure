import unittest

from report_facts import load_report_facts


class ReportConsistencyTests(unittest.TestCase):
    def test_report_facts_are_cross_file_consistent(self):
        facts = load_report_facts()
        self.assertEqual(facts["analysis_n"], 34)
        self.assertEqual(facts["analysis_groups"], 6)
        self.assertAlmostEqual(facts["fe_slope"], facts["permutation_observed"], places=12)
        self.assertEqual(facts["propagation_draws"], 1000)
        self.assertEqual(set(facts["scale_by_radius"]), {10, 25, 50})
        self.assertEqual(facts["influence_subgroup_omissions"], 6)
        self.assertEqual(facts["influence_scale_omissions"], 8)


if __name__ == "__main__":
    unittest.main()
