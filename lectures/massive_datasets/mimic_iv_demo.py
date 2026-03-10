# file: mimic_iv_demo.py
import pandas as pd, pyarrow.parquet as pq
admissions = pq.read_table('mimic-iv/icu/admissions.parquet').to_pandas()
one_stay   = admissions[admissions['hadm_id']==20017365]
print(one_stay[['subject_id','intime','outtime','los']])