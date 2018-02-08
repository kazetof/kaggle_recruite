from recruit import datasets
from recruit.utils import dataframe_utils as dfu
from recruit.plot import plot_data

d = datasets.RecruitDatasets()
avh = dfu.AirVisitHandler(d.air_visit)
index_list = [10]
plot_data.plot_air_visit(avh, index_list)