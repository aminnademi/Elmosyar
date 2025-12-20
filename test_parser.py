import unittest
import pandas as pd
from parser import Parser

class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = Parser("unit_tests.json")
        self.df = self.parser.run()

    def test_valid_message(self):
        """Parser should correctly extract all fields from a valid message"""
        row = self.df.iloc[0]
        # Check main fields
        self.assertIsNotNone(row["date"], "date should not be None")
        self.assertIsNotNone(row["term"], "term should not be None")
        self.assertIsNotNone(row["professor_name_raw"], "professor_name_raw should not be None")
        self.assertIsNotNone(row["department"], "department should not be None")
        self.assertIsNotNone(row["course_name"], "course_name should not be None")
        # Check ratings
        for col in self.parser.rating_lbl.keys():
            self.assertIsNotNone(row[col], f"{col} should not be None")
        # Check grading, attendance & comment
        self.assertIsNotNone(row["grading_status_raw"], "grading_status_raw should not be None")
        self.assertIsNotNone(row["attendance_status_raw"], "attendance_status_raw should not be None")
        self.assertIsNotNone(row["comment_text"], "comment_text should not be None")
        self.assertIsNone(row["parse_error"], "parse_error should be None for full message")

    def test_invalid_date(self):
        """Invalid date should appear in parse_error"""
        row = self.df.iloc[1]
        self.assertIn(self.parser.columns.index("date"), row["parse_error"])

    def test_missing_department_ratings(self):
        """If department & all ratings missing, parse_error should include all related indexes"""
        row = self.df.iloc[2]
        missings = [self.parser.columns.index("department")] + [self.parser.columns.index(c) for c in self.parser.rating_lbl]
        self.assertTrue(all(idx in row["parse_error"] for idx in missings))

    def test_full_ratings_and_term(self):
        """Row with all ratings and term should have parse_error=None & those values correctly extracted"""
        row = self.df.iloc[3]
        self.assertIsNone(row["parse_error"])
        for col in self.parser.rating_lbl:
            self.assertEqual(row[col], 10)
        self.assertEqual(row["term"], "تابستان 1400")

    def test_zero_ratings(self):
        """Row with all zero ratings should parse correctly"""
        row = self.df.iloc[4]
        for col in self.parser.rating_lbl:
            self.assertEqual(row[col], 0)
        self.assertIsNone(row["parse_error"])

if __name__ == "__main__":
    unittest.main()