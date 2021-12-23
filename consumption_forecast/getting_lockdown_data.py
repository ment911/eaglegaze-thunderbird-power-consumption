import os
import ast
import urllib.error

import psycopg2
import pandas as pd
import numpy as np
from eaglegaze_common.common_utils import insert_into_table
import datetime
from decouple import config as envs
from sqlalchemy import create_engine

engine = create_engine(envs('ALCHEMY_CONNECTION', cast=str))
con = engine.raw_connection()
cur = con.cursor()


def LockdownEU():
    cur = con.cursor()

    # SELECT MAX DATE TIME
    cur.execute('SELECT last_date FROM prime.lockdown_data_eu LIMIT 1')
    last_date = cur.fetchall()[0][0]

    # MAKE A DATE_TIME RANGE

    date_range = pd.date_range(start=last_date, end=datetime.datetime.now().date(), freq='D').map(
        lambda x: x.strftime('%Y-%m-%d')).tolist()

    url = 'https://www.ecdc.europa.eu/sites/default/files/documents/response_graphs_data_'
    dates = []
    for date in date_range:
        try:
            ld = pd.read_csv(f"{url}{date}.csv")
            dates.append(date)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                pass
    ld = pd.read_csv(f"{url}{max(dates)}.csv")
    cur.execute('DELETE FROM prime.lockdown_data_eu')
    con.commit()

    ld['date_start'] = pd.to_datetime(ld['date_start'])
    ld['date_end'] = pd.to_datetime(ld['date_end'])
    ld['date_end'] = ld['date_end'].fillna(np.nan).replace([np.nan], [None])
    ld['last_date'] = pd.to_datetime(max(dates))

    insert_into_table(ld, 'prime', 'lockdown_data_eu', primary_key=False,
                      custom_print='Lockdown EU data has been inserted')

if __name__ == '__main__':
    LockdownEU()

# def subset_sum(numbers, target, partial=[]):
#     s = sum(partial)
#
#     # check if the partial sum is equals to target
#     if s == target:
#         print ("sum(%s)=%s" % (partial, target))
#     if s >= target:
#         return  # if we reach the number why bother to continue
#
#     for i in range(len(numbers)):
#         n = numbers[i]
#         remaining = numbers[i + 1:]
#         subset_sum(remaining, target, partial + [n])
#
#
# if __name__ == "__main__":
#     subset_sum([3, 9, 8, 4, 5, 7, 10], 15)

