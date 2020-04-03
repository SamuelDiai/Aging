from .arterial_stiffness_processing import read_arterial_stiffness_data
from .blood_pressure_processing import read_blood_pressure_data
from .ecg_processing import read_ecg_at_rest_data
from .spirometry_processing import read_spirometry_data
import pandas as pd


def read_arterial_and_bp_data(**kwargs):
    df_1 = read_arterial_stiffness_data(**kwargs)
    df_2 = read_blood_pressure_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex'])
    return df_1.join(df_2, how = 'inner')

def read_cardiac_data(**kwargs):
    df_1 = read_arterial_stiffness_data(**kwargs)
    df_2 = read_blood_pressure_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex'])
    res = df_1.join(df_2, how = 'inner')
    df_3 = read_ecg_at_rest_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex'])
    return res.join(df_3, how = 'inner')

def read_spiro_and_arterial_and_bp_data(**kwargs):
    df_1 = read_arterial_stiffness_data(**kwargs)
    df_2 = read_blood_pressure_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex'])
    res = df_1.join(df_2, how = 'inner')
    df_3 = read_spirometry_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex'])
    return res.join(df_3, how = 'inner')
