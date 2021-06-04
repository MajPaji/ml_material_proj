# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:17:48 2021

@author: Dell
"""

import pandas as pd
import os





path = r'C:\Users\Dell\Documents\ml_material_proj'
df_raw = pd.read_csv(os.path.join(path, r'glassdoor_jobs.csv'))

# parsing of job description (python , etc)

# salary parsing

df = df_raw[df_raw['Salary Estimate'] != '-1']

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('$', '').replace('K', ''))

df['hourely'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour', ''))
min_employer_provided = min_hr.apply(lambda x: x.replace('employer provided salary:', ''))

df['min_salary'] = min_employer_provided.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_employer_provided.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

# company name text only

df['company_name_simplified'] = df.apply(
    lambda x: x['Company Name'].strip() if x.Rating == -1 else x['Company Name'].strip()[:-3], axis = 1)

# state field

df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])


df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

# age of company

df['age'] = df['Founded'].apply(lambda x: 2021 - x if x != -1 else -1)

# parsing of job description (python , etc)

# python

df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

# r studio

df['r_studio'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)

# spark

df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

# aws

df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

# excel

df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


df = df.drop('Unnamed: 0', axis = 1)
df = df.drop('Unnamed: 0.1', axis = 1)

df.to_csv(os.path.join(path, 'glassdoor_jobs_cleaned.csv'))
