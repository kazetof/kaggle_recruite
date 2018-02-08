import unittest

from recruit.utils import dataframe_utils as dfu
from recruit import datasets

import pandas as pd

class TestBase(unittest.TestCase):
    """
        base class
    """
    # loadig d at begging only once to speed up test.
    print("loading d")
    d = datasets.RecruitDatasets()
    unique_id = "air_877f79706adbfb06"


class TestAirReserveHandler(TestBase):
    """
    develop test example
        from tests.test_dataframe_utils import TestAirReserveHandler
        test = TestAirReserveHandler()
        test.setUpClass()
        unique_id = test.arh.get_unique_id()
        df_subset = test.arh.get_df_subset(unique_id[0])
    """
    @classmethod
    def setUpClass(cls):
        cls.arh = dfu.AirReserveHandler(cls.d.air_reserve)
        cls.unique_id_list = cls.arh.get_unique_id()

    def test_get_df_subset(self):
        for unique_id in self.unique_id_list:
            print(f"--- testing {unique_id} ---")
            df_subset = self.arh.get_df_subset(unique_id)
            print(f"len(df_subset) : {len(df_subset)}")
            self.assertIsInstance(df_subset, pd.core.frame.DataFrame)

    def test_get_visitdate_visiters_num_df(self):
        for unique_id in self.unique_id_list:
            print(f"--- testing {unique_id} ---")
            df_subset = self.arh.get_df_subset(unique_id)
            visit_date_num = self.arh.get_visitdate_visiters_num_df(df_subset)
            print(f"len(visit_date_num) : {len(visit_date_num)}")
            self.assertIsInstance(visit_date_num, pd.core.frame.DataFrame, msg=f"{type(visit_date_num)}")


class TestAirVisitHandler(TestBase):
    @classmethod
    def setUpClass(cls):
        cls.avh = dfu.AirVisitHandler(cls.d.air_visit)
        cls.unique_id_list = cls.avh.get_unique_id()

    @unittest.skip("test_get_df_subset has skipped.") # timeconsuming
    def test_get_df_subset(self):
        for unique_id in self.unique_id_list:
            print(f"--- testing {unique_id} ---")
            df_subset = self.avh.get_df_subset(unique_id)
            print(f"len(df_subset) : {len(df_subset)}")
            self.assertIsInstance(df_subset, pd.core.frame.DataFrame)
            self.assertEqual(pd.Timestamp("2017-04-22"), df_subset.visit_date.iloc[-1], msg=f"ID : {unique_id}")


if __name__ == '__main__':
    unittest.main(exit=False) # for REPL
    # unittest.main()
