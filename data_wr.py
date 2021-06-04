# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:33:03 2021

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('https://raw.githubusercontent.com/PlayingNumbers/ds_salary_proj/master/glassdoor_jobs.csv')

path = r'C:\Users\Dell\Documents\ml_material_proj'
df.to_csv(os.path.join(path, r'glassdoor_jobs.csv'))

