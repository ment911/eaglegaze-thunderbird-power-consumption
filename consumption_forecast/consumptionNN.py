import ast
import datetime
import os

from eaglegaze_common.thunderbird.thunderattr import *
from eaglegaze_common.thunderbird.thunder_utils import ThunderbirdUtils
import numpy as np
import pandas as pd
import psycopg2
import warnings
from dotenv import load_dotenv, find_dotenv
from eaglegaze_common.common_utils import insert_into_table, dublicated_hour, reduce_memory_usage, \
    resolve_psycopg2_programming_error
from getting_lockdown_data import LockdownEU

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())
DB_PARAMS = ast.literal_eval(os.environ["DB_PARAMS"])
con = psycopg2.connect(**DB_PARAMS)
cur = con.cursor()


class ConsumptionNN:
    '''All used tables'''
    balance_table = 'power_balance_entsoe_bi'
    sunset_sunrise = 'sunset_sunrise'
    openweather_hist = 'openweather_hist'
    calendar = 'working_calendar'
    openweather = 'openweather'
    pmi = 'pmi_data'
    unemployment_data = 'unemployment_data'
    raw_data_weekahead = 'thunderbird_raw_data_hu_consumption_weekahead'
    raw_data = 'thunderbird_raw_data_hu_consumption_lt'
    sun_inclination_data = 'sun_inclination'
    actual_forecast = 'openweatherhourlyfcactual'
    average_weather_data = 'openweatheraverage'
    trend_weather_data = 'openweatheraverageyears'
    raw_data_longterm = 'thunderbird_raw_data_hu_consumption_longterm'

    def __init__(self, country_code, local_time=True):
        self.country_code = country_code
        self.local_time = local_time

    def extract_consumption_data(self):
        cur.execute(f"SELECT date_time, value FROM bi.{ConsumptionNN.balance_table}"
                    f" WHERE country_code = '{self.country_code}' and id = 22 ORDER BY date_time")
        df_consumption = pd.DataFrame(cur.fetchall()).rename(columns={0: 'date_time', 1: 'consumption'})
        df_consumption = reduce_memory_usage(df_consumption)
        df_consumption['consumption'] = df_consumption['consumption'].astype(float)
        df_consumption['consumption'] = df_consumption['consumption'].replace(0, np.nan)
        df_consumption = ThunderbirdUtils().check_missing_values(frame=df_consumption, target_column='consumption')
        df_consumption = ThunderbirdUtils().z_test(frame=df_consumption, max_possible_std=3.2,
                                                   target_column='consumption')
        return df_consumption

    def get_avg_consumption(self, df, working_day_column='working_day'):

        df_working = df[df[working_day_column] == 1].dropna()
        df_non_working = df[df[working_day_column] == 0].dropna()

        values_working = pd.pivot_table(df_working.dropna(), values='consumption', aggfunc=np.mean,
                                        index=df_working['date_time'].dt.hour.astype(str) + '_' + df_working[
                                            'date_time'].dt.week.astype(str)).reset_index().rename(
            columns={'date_time': 'idx', 'consumption': 'avg_consumption'})

        values_non_working = pd.pivot_table(df_non_working.dropna(), values='consumption', aggfunc=np.mean,
                                            index=df_non_working['date_time'].dt.hour.astype(str) + '_' +
                                                  df_non_working['date_time'].dt.week.astype(str)).reset_index().rename(
            columns={'date_time': 'idx', 'consumption': 'avg_consumption'})

        df_w = df[df[working_day_column] == 1]
        df_nw = df[df[working_day_column] == 0]

        df_w['idx'] = df_w['date_time'].dt.hour.astype(str) + '_' + df_w['date_time'].dt.week.astype(str)
        df_w = pd.merge(df_w, values_working, on='idx', how='outer').drop(columns=['idx'])

        df_nw['idx'] = df_nw['date_time'].dt.hour.astype(str) + '_' + df_nw['date_time'].dt.week.astype(str)
        df_nw = pd.merge(df_nw, values_non_working, on='idx', how='outer').drop(columns=['idx'])

        df = pd.concat([df_w, df_nw])

        if len(df[df['avg_consumption'].isna()]) > 0:
            for index, row in df[df['avg_consumption'].isna()].iterrows():
                df.loc[df['date_time'] == row['date_time'], 'avg_consumption'] = df[
                    (df['date_time'].dt.month == row['date_time'].month) & (
                            df['date_time'].dt.hour == row['date_time'].hour) & (
                            df[working_day_column] == row[working_day_column])]['consumption'].mean()
        return df

    def lockdown_data(self, df):
        LockdownEU()

        cur.execute(f"SELECT сountry_name FROM bi.countries WHERE iso_сode = '{self.country_code}';")
        c_name = cur.fetchall()[0][0]

        if self.country_code == 'CZ':
            c_name = 'Czechia'

        cur.execute(f"SELECT * FROM prime.lockdown_data_eu WHERE country = '{c_name}'")
        ld = pd.DataFrame(cur.fetchall())
        ld.columns = [d[0] for d in cur.description]
        strict_ld = ld[ld['response_measure'] == 'StayHomeOrder']
        soft_ld = ld[ld['response_measure'] == 'StayHomeOrderPartial']

        if len(strict_ld) > 0:
            df['st_ld'] = 0
            for id in np.arange(len(strict_ld)):
                df.loc[(df['date_time'] >= pd.to_datetime(strict_ld.iloc[id]['date_start'])) & (
                        df['date_time'] <= pd.to_datetime(strict_ld.iloc[id]['date_end'])), 'st_ld'] = 1
        else:
            df['st_ld'] = 0

        df['st_ld'] = df['st_ld'].fillna(0)

        if len(soft_ld) > 1:
            df['sf_ld'] = 0
            for id in np.arange(len(soft_ld)):
                df.loc[(df['date_time'] >= pd.to_datetime(soft_ld.iloc[id]['date_start'])) & (
                        df['date_time'] <= pd.to_datetime(soft_ld.iloc[id]['date_end'])),'sf_ld'] = 1
        else:
            df['sf_ld'] = 0

        df['sf_ld'] = df['sf_ld'].fillna(0)

        return df

    def add_lag(self, min_lag=48, df=None, target_column='consumption'):
        """Add a time-lag"""
        df = df.sort_values('date_time', ascending=True)
        frame = df.copy()
        if min_lag == 48:
            frame['c_48'] = frame[target_column].shift(48)
            frame['c_96'] = frame[target_column].shift(96)
            frame['c_168'] = frame[target_column].shift(168)
            frame['c_336'] = frame[target_column].shift(336)
            frame['c_504'] = frame[target_column].shift(504)
            frame['c_672'] = frame[target_column].shift(672)
            frame = frame[~frame['c_672'].isna()]
            frame = frame[~frame['c_504'].isna()]
            frame = frame[~frame['c_336'].isna()]
            frame = frame[~frame['c_168'].isna()]
            frame = frame[~frame['c_96'].isna()]
            frame = frame[~frame['c_48'].isna()]
        elif min_lag == 168:
            frame['c_168'] = frame[target_column].shift(168)
            frame['c_336'] = frame[target_column].shift(336)
            frame['c_504'] = frame[target_column].shift(504)
            frame['c_672'] = frame[target_column].shift(672)
            frame = frame[~frame['c_672'].isna()]
            frame = frame[~frame['c_504'].isna()]
            frame = frame[~frame['c_336'].isna()]
            frame = frame[~frame['c_168'].isna()]
        if target_column == 'consumption':
            frame = frame[~((frame['consumption'].isna()) & (frame['date_time'] < '2021-06-01'))]
        return frame

    def full_calendar(self):
        df_calendar = ThunderbirdUtils(country_code=self.country_code).extract_calendar_data()

        day_before = []
        day_in_the_middle = []
        friday = []
        holiday_indexes = df_calendar[df_calendar['working_day'] == 0].index.tolist()
        for index, row in df_calendar[df_calendar['working_day'] == 0].iterrows():
            if ((index - 2) in holiday_indexes) and ((index - 3) in holiday_indexes):
                day_before.append(index-4)
            elif (index - 2) in holiday_indexes:
                day_before.append(index-3)
            elif (index + 1 not in  holiday_indexes) and (index - 1 not in holiday_indexes):
                day_before.append(index - 1)
            elif (index - 1) not in holiday_indexes:
                friday.append(index-1)
        for index, row in df_calendar[df_calendar['working_day'] == 1].iterrows():
            if ((index - 1) in holiday_indexes) and ((index + 1) in holiday_indexes):
                day_in_the_middle.append(index)
        dl = list(set([*day_before, *day_in_the_middle, *friday]))
        holiday_concerned = list(set([*day_before, *day_in_the_middle]))
        for idx in dl:
            df_calendar.loc[idx, 'day_before'] = 1
        for idx in holiday_concerned:
            df_calendar.loc[idx, 'holiday_concerned'] = 1
        df_calendar['day_before'] = df_calendar['day_before'].fillna(0)
        df_calendar['holiday_concerned'] = df_calendar['holiday_concerned'].fillna(0)
        return df_calendar

    def get_a_solar_rooftop_forecast(self, scenario=1):
        # Get market id
        cur.execute(f"SELECT m_id FROM im.im_market LEFT JOIN bi.countries ON m_sid1 = id  WHERE iso_сode = '"
                    f"{self.country_code}';")
        m_id = cur.fetchall()[0][0]
        # Get backtest
        if self.local_time:
            cur.execute(f"SELECT mfc_datetime_local, mfc_val_3 FROM im.im_markets_forecast_calc "
                        f"WHERE mfc_scenario = 4 AND mfc_market_id = {m_id} "
                        f"AND mfc_commodity_id = 1 AND mfc_microservice_id = 30")
            backtest = pd.DataFrame(cur.fetchall())
        else:
            cur.execute(f"SELECT mfc_datetime_utc, mfc_val_3 FROM im.im_markets_forecast_calc "
                        f"WHERE mfc_scenario = 4 AND mfc_market_id = {m_id} "
                        f"AND mfc_commodity_id = 1 AND mfc_microservice_id = 30")
            backtest = pd.DataFrame(cur.fetchall())
        if len(backtest):
            backtest.columns = [d[0] for d in cur.description]
            # Get forecast
            if self.local_time:
                cur.execute(f"SELECT mfc_datetime_local, mfc_val_3 FROM im.im_markets_forecast_calc "
                            f"WHERE mfc_scenario = {scenario} AND mfc_market_id = {m_id} "
                            f"AND mfc_commodity_id = 1 AND mfc_microservice_id = 30")
                forecast = pd.DataFrame(cur.fetchall())
            else:
                cur.execute(f"SELECT mfc_datetime_utc, mfc_val_3 FROM im.im_markets_forecast_calc "
                            f"WHERE mfc_scenario = {scenario} AND mfc_market_id = {m_id} "
                            f"AND mfc_commodity_id = 1 AND mfc_microservice_id = 30")
                forecast = pd.DataFrame(cur.fetchall())
            forecast.columns = [d[0] for d in cur.description]
            forecast = forecast[forecast['mfc_datetime_local'] > backtest['mfc_datetime_local'].max()]
            df = pd.concat([backtest, forecast])
            df = reduce_memory_usage(df)
            if self.local_time:
                df = dublicated_hour(df=df, subset=['mfc_datetime_local'], date_time_column='mfc_datetime_local').rename(
                    columns={'mfc_datetime_local': 'date_time',
                             'mfc_val_3': 'prediction'}
                )
            else:
                df = df.rename(columns={'mfc_datetime_utc': 'date_time', 'mfc_val_3': 'prediction'})
            return df
        else:
            print(f"\n \n \n No backtest had been done yet for {self.country_code}, sin inclination values will be "
                  f"inserted as a proxy!!!!!! \n \n \n")
            return(f"\n \n \n No backtest had been done yet for {self.country_code}, sin inclination values will be "
                  f"inserted as a proxy!!!!!! \n \n \n")

    def collect_data_for_2d_forecast(self):

        df_consumption = self.extract_consumption_data()
        df_weather = ThunderbirdUtils.Weather(country_code=self.country_code,
                                              local_time=self.local_time).weather_forecast()
        df_calendar = self.full_calendar()
        df_sun = ThunderbirdUtils(country_code=self.country_code).extract_set_rise_data()
        sun_inc = ThunderbirdUtils(country_code=self.country_code).sun_inclination()

        ####
        df_weather = df_weather[df_weather['date_time'] > df_consumption['date_time'].min()]
        df_consumption = pd.merge(df_weather, df_consumption, on='date_time', how='outer')
        df_consumption.dropna(inplace=True, thresh=4)

        ####
        df_consumption['rain'] = df_consumption['rain'].fillna(0)
        df_consumption['snow'] = df_consumption['snow'].fillna(0)

        # Add timelag

        df_consumption = self.add_lag(min_lag=48, df=df_consumption)

        # Adding calendar
        df_consumption = pd.merge(df_consumption, df_calendar, left_on=df_consumption['date_time'].dt.date,
                                  right_on='date', how='outer').drop(
            columns=['date'])
        # df_consumption = pd.merge(df_consumption, df_weather, on='date_time')
        df_consumption.fillna(np.nan, inplace=True)
        df_consumption.dropna(thresh=df_consumption.shape[1] - 1, inplace=True)
        # df_consumption.dropna(thresh=6, inplace=True)

        cdh_hdh = self.cdh_hdh(two_days=True)
        df_consumption = pd.merge(df_consumption, cdh_hdh, on='date_time').drop(
            columns=['temperature_x']).rename(columns={'temperature_y': 'temperature'})
        # Adding dummise to df
        dummies = pd.get_dummies(df_consumption['date_time'].dt.hour, prefix='d')
        df = pd.concat([df_consumption, dummies], axis=1)
        # df.fillna(0, inplace=True)

        # Adding sun hours
        df = pd.merge(df, df_sun, left_on=df['date_time'].dt.date, right_on=df_sun['date_time'].dt.date,
                      how='outer').drop(columns=['date_time_y', 'key_0']).rename(columns={'date_time_x': 'date_time'})
        df = df[~df['d_1'].isna()]

        df = ThunderbirdUtils(country_code=self.country_code).is_sunny_hour(df)

        # Add sun inclination
        df = pd.merge(df, sun_inc, left_on=df['date_time'].dt.date.astype(str) + '_' + df['date_time'].dt.hour.astype(
            str),
                      right_on=sun_inc['date_time'].dt.date.astype(str) + '_' + sun_inc['date_time'].dt.hour.astype(
                          str),
                      how='outer').drop(columns=['date_time_y', 'key_0']).rename(
            columns={'date_time_x': 'date_time'})
        df = df[~df['d_1'].isna()]

        """PUT THE MFC HERE"""
        if self.country_code in ConsumptionForecast.no_solar_countries.value:
            df['prediction'] = df['sin'].fillna(0)
        else:
            rooftop = self.get_a_solar_rooftop_forecast()
            if isinstance(rooftop, pd.DataFrame):
                df = pd.merge(df, rooftop, on='date_time', how='outer').dropna(thresh=df.shape[1]-1)
                df['prediction'] = df['prediction'].fillna(0)
            elif isinstance(rooftop, str):
                df['prediction'] = df['sin'].fillna(0)
        df.drop(columns=['sin', 'clouds'], inplace=True)
        df.dropna(thresh=df.shape[1] - 1, inplace=True)

        df = self.lockdown_data(df)

        df['Wx'] = df['wind_speed'].astype(float) * np.cos(df['wind_deg'].astype(float) * np.pi / 180)
        df['Wy'] = df['wind_speed'].astype(float) * np.sin(df['wind_deg'].astype(float) * np.pi / 180)
        df.drop(columns=['wind_speed', 'wind_deg'], inplace=True)

        timestamp_s = df['date_time'].map(datetime.datetime.timestamp)
        day = 24 * 60 * 60
        df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))

        df_dw = pd.get_dummies(df['date_time'].dt.weekday)
        df_dw.columns = ['dw_' + str(i + 1) for i in df_dw.columns.tolist()]
        df = pd.concat([df, df_dw], axis=1)

        df.drop(columns=['day_week'], inplace=True)
        df.drop_duplicates(subset=['date_time'], inplace=True)

        df.dropna(thresh=df.shape[1] - 1, inplace=True)

        df['country_code'] = self.country_code

        try:
            insert_into_table(df, 'prime', ConsumptionNN.raw_data)
        except psycopg2.ProgrammingError:
            df = resolve_psycopg2_programming_error(df)
            insert_into_table(df, 'prime', ConsumptionNN.raw_data)


    def collect_data_for_weekahead_forecast(self):

        df_consumption = self.extract_consumption_data()
        df_weather = ThunderbirdUtils.Weather(country_code=self.country_code,
                                              local_time=self.local_time).weather_interpolation()
        df_calendar = self.full_calendar()
        df_sun = ThunderbirdUtils(country_code=self.country_code).extract_set_rise_data()
        sun_inc = ThunderbirdUtils(country_code=self.country_code).sun_inclination()

        ####
        df_weather = df_weather[df_weather['date_time'] > df_consumption['date_time'].min()]
        df_consumption = pd.merge(df_weather, df_consumption, on='date_time', how='outer')
        df_consumption.dropna(inplace=True, thresh=df_consumption.shape[1] - 1)
        ####

        # Add timelag
        df_consumption = self.add_lag(min_lag=168, df=df_consumption)

        # Adding calendar
        df_consumption = pd.merge(df_consumption, df_calendar, left_on=df_consumption['date_time'].dt.date,
                                  right_on='date', how='outer').drop(columns=['date'])
        # df_consumption = pd.merge(df_consumption, df_weather, on='date_time')
        df_consumption.fillna(np.nan, inplace=True)
        df_consumption.dropna(thresh=df_consumption.shape[1] - 1, inplace=True)

        cdh_hdh = self.cdh_hdh(weekahead=True)
        df_consumption = pd.merge(df_consumption, cdh_hdh, on='date_time').drop(
            columns=['temperature_x']).rename(columns={'temperature_y': 'temperature'})

        # Adding dummise to df
        dummies = pd.get_dummies(df_consumption['date_time'].dt.hour, prefix='d')
        df = pd.concat([df_consumption, dummies], axis=1)
        # df.fillna(0, inplace=True)

        # Adding sun hours
        ''' New method '''
        df = pd.merge(df, df_sun, left_on=df['date_time'].dt.date, right_on=df_sun['date_time'].dt.date,
                      how='outer').drop(columns=['date_time_y', 'key_0']).rename(columns={'date_time_x': 'date_time'})
        df = df[~df['d_1'].isna()]

        df = ThunderbirdUtils(country_code=self.country_code).is_sunny_hour(df)

        # Add sun inclination
        df = pd.merge(df, sun_inc, left_on=df['date_time'].dt.date.astype(str) + '_' + df['date_time'].dt.hour.astype(
            str),
                      right_on=sun_inc['date_time'].dt.date.astype(str) + '_' + sun_inc['date_time'].dt.hour.astype(
                          str),
                      how='outer').drop(columns=['date_time_y', 'key_0']).rename(
            columns={'date_time_x': 'date_time'})
        df = df[~df['d_1'].isna()]
        df['sin'] = df['sin'].fillna(0)

        """PUT THE MFC HERE"""
        if self.country_code in ConsumptionForecast.no_solar_countries.value:
            df['prediction'] = df['sin'].fillna(0)
        else:
            rooftop = self.get_a_solar_rooftop_forecast()
            if isinstance(rooftop, pd.DataFrame):
                df = pd.merge(df, rooftop, on='date_time', how='outer').dropna(thresh=df.shape[1]-1)
                df['prediction'] = df['prediction'].fillna(0)
            elif isinstance(rooftop, str):
                df['prediction'] = df['sin'].fillna(0)
        df.drop(columns=['sin', 'clouds'], inplace=True)
        df.dropna(thresh=df.shape[1] - 1, inplace=True)

        df.dropna(thresh=df.shape[1] - 1, inplace=True)
        df = self.lockdown_data(df)

        timestamp_s = df['date_time'].map(datetime.datetime.timestamp)
        day = 24 * 60 * 60
        df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))

        df_dw = pd.get_dummies(df['date_time'].dt.weekday)
        df_dw.columns = ['dw_' + str(i + 1) for i in df_dw.columns.tolist()]
        df = pd.concat([df, df_dw], axis=1)

        df.drop(columns=['day_week'], inplace=True)
        df.drop_duplicates(subset=['date_time'], inplace=True)
        df.dropna(thresh=df.shape[1] - 1, inplace=True)
        df['country_code'] = self.country_code

        try:
            insert_into_table(df, 'prime', ConsumptionNN.raw_data)
        except psycopg2.ProgrammingError:
            df = resolve_psycopg2_programming_error(df)
            insert_into_table(df, 'prime', ConsumptionNN.raw_data)

    def collect_data_for_longterm_forecast(self):

        df_consumption = self.extract_consumption_data()
        df_weather = ThunderbirdUtils.Weather(country_code=self.country_code,
                                              local_time=self.local_time).extract_weather_trend_data(till=2030)
        df_calendar = self.full_calendar()
        df_sun = ThunderbirdUtils(country_code=self.country_code).extract_set_rise_data()
        sun_inc = ThunderbirdUtils(country_code=self.country_code).sun_inclination()

        # Adding weather
        df_weather = df_weather[df_weather['date_time'] > df_consumption['date_time'].min()]
        df_consumption = pd.merge(df_weather, df_consumption, on='date_time', how='outer')
        df_consumption['rain'] = df_consumption['rain'].fillna(0)
        df_consumption['snow'] = df_consumption['snow'].fillna(0)

        df_consumption.dropna(inplace=True, thresh=df_consumption.shape[1] - 1)
        ####

        # Adding calendar
        df_consumption = pd.merge(df_consumption, df_calendar, left_on=df_consumption['date_time'].dt.date,
                                  right_on='date', how='outer').drop(columns=['date'])
        df_consumption.drop_duplicates(subset=['date_time'], inplace=True)
        # df_consumption = pd.merge(df_consumption, df_weather, on='date_time')
        df_consumption.fillna(np.nan, inplace=True)
        df_consumption.dropna(thresh=df_consumption.shape[1] - 1, inplace=True)

        cdh_hdh = self.cdh_hdh(longterm=True)
        df_consumption = pd.merge(df_consumption, cdh_hdh, on='date_time').drop(
            columns=['temperature_x']).rename(columns={'temperature_y': 'temperature'})

        df_consumption = df_consumption.sort_values('date_time')
        # Adding dummise to df
        dummies = pd.get_dummies(df_consumption['date_time'].dt.hour, prefix='d')
        df = pd.concat([df_consumption, dummies], axis=1)

        # Adding sun hours
        ''' New method '''
        df = pd.merge(df, df_sun, left_on=df['date_time'].dt.date, right_on=df_sun['date_time'].dt.date,
                      how='outer').drop(columns=['date_time_y', 'key_0']).rename(columns={'date_time_x': 'date_time'})
        df = df[~df['d_1'].isna()]

        df = ThunderbirdUtils(country_code=self.country_code).is_sunny_hour(df)

        # Add sun inclination
        df = pd.merge(df, sun_inc, left_on=df['date_time'].dt.date.astype(str) + '_' + df['date_time'].dt.hour.astype(
            str),
                      right_on=sun_inc['date_time'].dt.date.astype(str) + '_' + sun_inc['date_time'].dt.hour.astype(
                          str),
                      how='outer').drop(columns=['date_time_y', 'key_0']).rename(
            columns={'date_time_x': 'date_time'})
        df = df[~df['d_1'].isna()]

        """PUT THE MFC HERE"""
        if self.country_code in ConsumptionForecast.no_solar_countries.value:
            df['prediction'] = df['sin'].fillna(0)
        else:
            rooftop = self.get_a_solar_rooftop_forecast()
            if isinstance(rooftop, pd.DataFrame):
                df = pd.merge(df, rooftop, on='date_time', how='outer').dropna(thresh=df.shape[1]-1)
                df['prediction'] = df['prediction'].fillna(0)
            elif isinstance(rooftop, str):
                df['prediction'] = df['sin'].fillna(0)
        df.drop(columns=['sin', 'clouds'], inplace=True)
        df.dropna(thresh=df.shape[1] - 1, inplace=True)

        # Adding lockdaown data
        df = self.lockdown_data(df)

        df['Wx'] = df['wind_speed'].astype(float) * np.cos(df['wind_deg'].astype(float) * np.pi / 180)
        df['Wy'] = df['wind_speed'].astype(float) * np.sin(df['wind_deg'].astype(float) * np.pi / 180)
        df.drop(columns=['wind_speed', 'wind_deg'], inplace=True)

        timestamp_s = df['date_time'].map(datetime.datetime.timestamp)
        day = 24 * 60 * 60
        df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))

        df_dw = pd.get_dummies(df['date_time'].dt.weekday)
        df_dw.columns = ['dw_' + str(i + 1) for i in df_dw.columns.tolist()]
        df = pd.concat([df, df_dw], axis=1)

        df.drop(columns=['day_week'], inplace=True)
        df.drop_duplicates(subset=['date_time'], inplace=True)

        df.dropna(thresh=df.shape[1] - 1, inplace=True)

        df = self.get_avg_consumption(df)
        df['country_code'] = self.country_code
        try:
            insert_into_table(df, 'prime', ConsumptionNN.raw_data)
        except psycopg2.ProgrammingError:
            df = resolve_psycopg2_programming_error(df)
            insert_into_table(df, 'prime', ConsumptionNN.raw_data)

    @staticmethod
    def find_t_bound(df, hour, is_working_day, root=3, verbose=False):
        """
        Function that finds the intersection of two linear equations.
        The point of intersection is a juncture where colling ends and heating begins.
        :param df: pd.DataFrame has to be provided
        :param hour: Which hour has to be analysed
        :param is_working_day: working day = 1, dayoff = 0
        :param root: the degree of polynom, either 2 or 3, 3 is better
        :param verbose: Default = False
        :return: Float number
        """

        y = df[(df['date_time'].dt.hour == hour) & (df['working_day'] == is_working_day)]['consumption'] * -1
        x = df[(df['date_time'].dt.hour == hour) & (df['working_day'] == is_working_day)]['temperature']
        if root == 2:
            z = np.polyfit(x, y, root)
            p = np.poly1d(z)
            t_bound = z[1] / (z[0] * root) * -1
        elif root == 3:
            z = np.polyfit(x, y, root)
            p = np.poly1d(z)
            coef = [z[0] * 3, z[1] * 2, z[2]]
            t_bound = np.roots(coef)[0]
        if t_bound < 0:
            t_bound = t_bound * -1
        if verbose:
            print('equation: ', p)
            print(f"For hour: '{hour}' \t temp bound is: {t_bound}")

        """To uncomment this staff import matplolib and add plot to the params of the function"""
        # if plot:
        #     temp = np.linspace(-10, 25, len(df_t['temp']))
        #     plt.rcParams["figure.figsize"] = (8, 6)
        #     plt.scatter(x, y, marker='o')
        #     plt.plot(temp, p(temp), "r--")
        #     plt.show()

        return t_bound

    def find_t_bound_adv(self):
        """
        Finds temperature bounds for a country provided in class __init__ method
        Return 2 dict concerned working and dayoff hours
        """
        df_consumption = self.extract_consumption_data()
        df_calendar = ThunderbirdUtils(country_code=self.country_code).extract_calendar_data()
        df_weather = ThunderbirdUtils.Weather(country_code=self.country_code,
                                              local_time=self.local_time).weather_forecast()

        df_weather = df_weather[df_weather['date_time'] > df_consumption['date_time'].min()]
        df_consumption = pd.merge(df_weather.drop(columns=['rain', 'snow']), df_consumption, on='date_time',
                                  how='outer')

        df_consumption = pd.merge(df_consumption, df_calendar, left_on=df_consumption['date_time'].dt.date,
                                  right_on='date', how='outer').drop(
            columns=['date'])
        df = df_consumption.dropna()

        dh = {}
        dh0 = {}
        for hour in list(np.arange(0, 24)):
            dh[hour] = self.find_t_bound(df, hour, 1)
            dh0[hour] = self.find_t_bound(df, hour, 0)

        return dh, dh0

    def cdh_hdh(self, two_days=False, weekahead=False, longterm=False):

        dh, dh0 = self.find_t_bound_adv()
        if two_days:
            df_weather, points = ThunderbirdUtils.Weather(country_code=self.country_code,
                                                  local_time=self.local_time, weighted=True).weather_forecast()
        elif weekahead:
            df_weather, points = ThunderbirdUtils.Weather(country_code=self.country_code,
                                                  local_time=self.local_time, weighted=True).weather_interpolation()
        elif longterm:
            df_weather, points = ThunderbirdUtils.Weather(country_code=self.country_code,
                                                  local_time=self.local_time, weighted=True).extract_weather_trend_data()
        weights = []
        for point in points:
            cur.execute(f"SELECT consumption_weight FROM bi.weatherpointsref WHERE point_name = '{point}'")
            weights.append(cur.fetchall()[0][0])
        df_weather = pd.DataFrame({'date_time': df_weather['date_time'],
                                   'temperature': np.dot(df_weather.iloc[:, 1:].values, np.array(weights))})
        df_calendar = ThunderbirdUtils(country_code=self.country_code).extract_calendar_data()
        df = pd.merge(df_weather, df_calendar, left_on=df_weather['date_time'].dt.date, right_on='date').drop(
            columns=['date']).reset_index(drop=True)
        for key, value in dh.items():
            df.loc[(df['temperature'] < value - 1) & (df['working_day'] == 1) & (df['date_time'].dt.hour == key), 'hdh'] \
                = (df['temperature'] - (value - 1)) * -1
            df.loc[(df['temperature'] > value + 1) & (df['working_day'] == 1) & (df['date_time'].dt.hour == key), 'cdh'] \
                = (df['temperature'] - (value + 1))
        for key, value in dh0.items():
            df.loc[(df['temperature'] < value - 1) & (df['working_day'] == 0) & (df['date_time'].dt.hour == key), 'hdh'] \
                = (df['temperature'] - (value - 1)) * -1
            df.loc[(df['temperature'] > value + 1) & (df['working_day'] == 0) & (df['date_time'].dt.hour == key), 'cdh'] \
                = (df['temperature'] - (value + 1))

        df['cdh'] = df['cdh'].fillna(0)
        df['hdh'] = df['hdh'].fillna(0)
        df.drop(columns=['day_week', 'working_day', 'holiday'], inplace=True)
        return df


if __name__ == '__main__':
    # ConsumptionNN().collect_data_for_longterm_forecast()
    # solar_prediction('HU').predict(solar_prediction('HU')._prepare_data(), 'value')
    # countries = ['SK', 'RO', 'PL', 'HU', 'CZ']
    countries = ['PL']
    for country in countries:
        #solarNN(country_code=country).gather_data_for_2d_forecast()
        ConsumptionNN(country_code=country).collect_data_for_2d_forecast()
        print(f'Data for 2d forecast has been extracted for {country}')
    # ConsumptionNN().collect_data_for_weekahead_forecast()
    # print('Data for weekahead forecast has been extracted')
