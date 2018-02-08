import numpy as np
import pandas as pd

def load_id_relation():
    id_relation = pd.read_csv("./data/store_id_relation.csv")
    return id_relation

def load_air_reserve():
    air_reserve = pd.read_csv("./data/air_reserve.csv")
    return air_reserve

def load_air_store_info():
    air_store = pd.read_csv("./data/air_store_info.csv")
    return air_store

def load_air_visit():
    air_visit = pd.read_csv("./data/air_visit_data.csv")
    return air_visit

def load_date_info():
    date_info = pd.read_csv("./data/date_info.csv")
    return date_info

def load_hpg_reserve():
    hpg_reserve = pd.read_csv("./data/hpg_reserve.csv")
    return hpg_reserve

def load_hpg_store_info():
    hpg_store = pd.read_csv("./data/hpg_store_info.csv")
    return hpg_store

def load_sample_submission():
    sample_sub = pd.read_csv("./data/sample_submission.csv")
    return sample_sub    

class RecruitDatasets(object):
    def __init__(self):
        self.id_relation = load_id_relation()
        self.air_reserve = load_air_reserve()
        self.air_store = load_air_store_info()
        self.air_visit = load_air_visit()
        self.date_info = load_date_info()
        self.hpg_reserve = load_hpg_reserve()
        self.hpg_store = load_hpg_store_info()
        self.sample_sub = load_sample_submission()




