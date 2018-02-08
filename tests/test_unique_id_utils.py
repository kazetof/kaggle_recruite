import unittest
import pandas as pd

from recruit.utils import unique_id_utils as uiu
from recruit import datasets


class TestUniqueIDHandler(unittest.TestCase):
    def setUp(self):
        data = datasets.RecruitDatasets()
        self.uih = uiu.UniqueIDHandler(data=data)
        self.unique_id_list = self.uih.get_sample_submission_unique_id_list()

    def test_get_dataframe(self):
        for unique_id in self.unique_id_list:
            uih = self.uih.set_unique_id(unique_id)
            df = self.uih.get_dataframe()
            print(df.head())
            print(df.tail())
            self.assertEqual(pd.Timestamp("2017-04-22"), df.visit_date.iloc[-1], msg=f"ID : {unique_id}")
            self.assertTrue(df.isnull().sum().sum() == 0, msg=f"ID : {unique_id}")


# from recruit.utils import dataframe_utils as dfu
# def check_error_id():
#     error_id = "air_083ddc520ea47e1e"
#     data = datasets.RecruitDatasets()
#     uih = uiu.UniqueIDHandler(unique_id=error_id, data=data)
#     df = uih.get_dataframe()

#     arh = dfu.AirReserveHandler(data.air_reserve)
#     avh = dfu.AirVisitHandler(data.air_visit)
#     air_visit_subset = avh.get_df_subset(error_id, completion=True)

if __name__ == '__main__':
    unittest.main(exit=False) # for REPL
    # unittest.main()



