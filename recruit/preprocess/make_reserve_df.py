from recruit.utils import dataframe_utils as dfu
from recruit import datasets

import pandas as pd
import numpy as np


d = datasets.RecruitDatasets()
avh = dfu.AirVisitHandler(d.air_visit)
arh = dfu.AirReserveHandler(d.air_reserve)

avh_id_list = avh.get_unique_id()
arh_id_list = arh.get_unique_id()

unique_id = avh_id_list[29]
avh_subset = avh.get_df_subset(unique_id, completion=True)
arh_subset = arh.get_df_subset(unique_id)
print("len(arh_subset) : ", len(arh_subset))

visitdate_visiters_num_df = arh.get_visitdate_visiters_num_df(arh_subset)
