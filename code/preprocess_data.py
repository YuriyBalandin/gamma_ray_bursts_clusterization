import pandas as pd
import numpy as np 
import os
from tqdm.notebook import tqdm
from numpy import trapz
from scipy.fft import  rfft, fftfreq


def preprocess_data():
    # getting burst names 
    burst_names = [i for i in os.listdir('../data/raw_data') if 'fth' in i]
    
    #padding
    len_burst = []
    nas = []
    for name in tqdm(burst_names):
        df = pd.read_feather(f'../data/raw_data/{name}')
        na = df.isna().sum()
        l = len(df)

        len_burst.append(l)
        nas.append(na)
        
    max_len = max(len_burst)

    for name in tqdm(burst_names):
        df = pd.read_feather(f'../data/raw_data/{name}')
        nzeros = max_len - len(df)
        if nzeros != 0:
            zeros = np.zeros((nzeros,6))
            zeros = pd.DataFrame(zeros)
            zeros.columns = df.columns
            df = pd.concat([df, zeros]).reset_index(drop=True)
            print(df)
            df.to_feather(f'../data/raw_data/{name}')
            
    # normalization
    
    data = []

    for name in tqdm(burst_names):
        df = pd.read_feather(f'../data/raw_data/{name}')

        y = df['15_25_keV']
        area = trapz(y)
        y0 = y/area

        y = df['25_50_keV']
        area = trapz(y)
        y1 = y/area

        y = df['50_100_keV']
        area = trapz(y)
        y2 = y/area

        y = df['100_350_keV']
        area = trapz(y)
        y3 = y/area

        y = list(y0) + list(y1) + list(y2) + list(y3)

        data.append(y)
    
    # discrete fourier transform 
    data = pd.DataFrame(data)

    a = rfft(data)
    a = pd.DataFrame(a)
    
    a = a.applymap(lambda x: abs(x))
    a = a.fillna(0)
    
    return a