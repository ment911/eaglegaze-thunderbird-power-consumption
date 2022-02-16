import ast
import datetime
import os

import pandas as pd
import psycopg2
from decouple import config as envs
from sqlalchemy import create_engine
# from eaglegaze_common.common_attr import Attributes as at
# import eaglegaze_common.logger as log
from eaglegaze_common.entsoe_configs import COUTRIES_SHIFTS
from eaglegaze_common.common_utils import insert_into_table, get_time_shift, substract_time_shift, get_iteration, \
    start_end_microservice_time_with_iteration, to_utc
from eaglegaze_common.thunderbird.get_data_and_predict import ThunderbirdPredict
from eaglegaze_common.thunderbird.nn_train_test import ThunderbirdTrain
from eaglegaze_common.thunderbird.scale_the_data import ThunderbirdScale
from eaglegaze_common.thunderbird.thunderattr import ConsumptionForecast
from eaglegaze_common.thunderbird.thunder_utils import ThunderbirdUtils, check_fact_scenario
from consumptionNN import ConsumptionNN
import pathlib
from getting_lockdown_data import LockdownEU

path_files = pathlib.Path(__file__).parent.resolve()
os.environ['SCALER_PATH'] = f"{path_files}/scalers/"
os.environ['MODEL_PATH']= f"{path_files}/models/"
MODEL_PATH = os.environ.get('MODEL_PATH')
engine = create_engine(envs('ALCHEMY_CONNECTION', cast=str))
con = engine.raw_connection()
cur = con.cursor()


class Thunder:

    def __init__(self, countries=None, backtest=False, scenario=1):
        self.countries = countries
        self.backtest = backtest
        self.mfc_microservice_id = 4
        if self.backtest:
            self.scenario = 5
        else:
            self.scenario = scenario

    def _get_fact_data(self, date_time_start, country):
        df = ConsumptionNN(country_code=country).extract_consumption_data()
        df = df[df['date_time'] >= date_time_start]
        df['consumption'] = df['consumption'] * -1
        return df

    def _create_fact_scenario(self, country, date_time_start, frame):
        df = self._get_fact_data(country=country, date_time_start=date_time_start)
        df['date_time'] = df['date_time'].apply(lambda x: x.replace(minute=0))
        df['date_time'] = to_utc(df['date_time'].tolist(), country_code=country)
        df = pd.merge(frame, df, left_on='mfc_datetime_utc', right_on='date_time').drop(columns=['date_time',
                                                                                                 'mfc_val_1']).rename(
            columns={'consumption': 'mfc_val_1'}
        )
        df = df.drop_duplicates(subset=['mfc_iteration', 'mfc_scenario', 'mfc_datetime_utc',
                                        'mfc_market_id', 'mfc_commodity_id', 'mfc_microservice_id'])
        if not check_fact_scenario(mfc_microservice_id=self.mfc_microservice_id, mfc_market_id=self.mfc_market_id):
            df['mfc_scenario'] = 4
            insert_into_table(df, 'im', 'im_markets_forecast_calc', primary_key=False,
                              constraint='unique_constraint')
            print(f" \n Consumption forecast for scenario 4 had been done \n")
        return df

    def _insert_different_scenarios(self, td_df, country):
        df = self._create_fact_scenario(country=country, date_time_start=td_df['mfc_datetime_local'].min(),
                                        frame=td_df)
        td_df = pd.concat([df, td_df[td_df['mfc_datetime_utc'] > df['mfc_datetime_utc'].max()]])
        td_df = td_df.drop_duplicates(subset=['mfc_datetime_utc'])
        td_df['mfc_scenario'] = 1
        insert_into_table(td_df, 'im', 'im_markets_forecast_calc', primary_key=False,
                          constraint='unique_constraint')
        td_df['mfc_scenario'] = 2
        insert_into_table(td_df, 'im', 'im_markets_forecast_calc', primary_key=False,
                          constraint='unique_constraint')
        td_df['mfc_scenario'] = 3
        insert_into_table(td_df, 'im', 'im_markets_forecast_calc', primary_key=False,
                          constraint='unique_constraint')
        print(f" \n Consumption forecast for scenarios 1, 2, 3 had been done \n")

    def consumption(self, training_day=5, two_days_ahead=True, weekahead=True, longterm=True, country=None, train=True):
        # Lets extract and rescale our data
        if self.scenario == 5:
            if two_days_ahead:
                ConsumptionNN(country_code=country).collect_data_for_2d_forecast()
                print(f'Data for 2d forecast has been extracted for {country}')
                ThunderbirdScale(raw_data=ConsumptionForecast.TwoDaysAhead.raw_data,
                                       scaled_data=ConsumptionForecast.TwoDaysAhead.scaled_data,
                                       scaled_y=ConsumptionForecast.TwoDaysAhead.scaled_y,
                                       scaler_name=f'{ConsumptionForecast.TwoDaysAhead.scaler_name}_{country}',
                                       y=ConsumptionForecast.y.value,
                                       country_code=country).rescale_data()
                print(f'Data for {country} for 2d forecast has been rescaled')
            if weekahead:
                ConsumptionNN(country_code=country).collect_data_for_weekahead_forecast()
                print(f'Data for WA forecast has been extracted for {country}')
                ThunderbirdScale(raw_data=ConsumptionForecast.Weekahead.raw_data,
                                       scaled_data=ConsumptionForecast.Weekahead.scaled_data,
                                       scaled_y=ConsumptionForecast.Weekahead.scaled_y,
                                       scaler_name=f'{ConsumptionForecast.Weekahead.scaler_name}_{country}',
                                       y=ConsumptionForecast.y.value,
                                       country_code=country).rescale_data()
                print(f'Data for {country} for WA forecast has been rescaled')
            if longterm:
                ConsumptionNN(country_code=country).collect_data_for_longterm_forecast()
                print(f'Data for WA forecast has been extracted for {country}')
                ThunderbirdScale(raw_data=ConsumptionForecast.Longterm.raw_data,
                                       scaled_data=ConsumptionForecast.Longterm.scaled_data,
                                       scaled_y=ConsumptionForecast.Longterm.scaled_y,
                                       scaler_name=f'{ConsumptionForecast.Longterm.scaler_name}_{country}',
                                       y=ConsumptionForecast.y.value,
                                       country_code=country).rescale_data()
                print(f'Data for {country} for longterm forecast has been rescaled')

        models_dirs = [f'{MODEL_PATH}/{ConsumptionForecast.TwoDaysAhead.directory}/{country}',
                       f'{MODEL_PATH}/{ConsumptionForecast.Weekahead.directory}/{country}',
                       f'{MODEL_PATH}/{ConsumptionForecast.Longterm.directory}/{country}']

        check_list = []
        for dirs in models_dirs: check_list.append(os.path.isdir(dirs))
        if False in check_list:
            training_day = datetime.datetime.now().weekday()

        if datetime.datetime.now().weekday() == training_day and train:
            if two_days_ahead:
                ThunderbirdTrain(scaled_data=ConsumptionForecast.TwoDaysAhead.scaled_data,
                                  model_path=f'{MODEL_PATH}/{ConsumptionForecast.TwoDaysAhead.directory}'
                                             f'/{country}',
                                  y_data=ConsumptionForecast.TwoDaysAhead.scaled_y,
                                  country_code=country).create_model(epochs=1000, batch_size=32, train_size=0.97,
                                                                     shuffle=True, patience=60, shuffle_on_train=True)
                print(f' \n 2DA consumption() model for {country} has been retrained \n')
            if weekahead:
                ThunderbirdTrain(scaled_data=ConsumptionForecast.Weekahead.scaled_data,
                                  model_path=f'{MODEL_PATH}/{ConsumptionForecast.Weekahead.directory}/{country}',
                                  y_data=ConsumptionForecast.Weekahead.scaled_y,
                                  country_code=country).create_model(epochs=1000, batch_size=32, train_size=0.97,
                                                                     shuffle=True, patience=60, shuffle_on_train=True)
                print(f' \n WA consumption() model for {country} has been retrained \n')
            if longterm:
                ThunderbirdTrain(scaled_data=ConsumptionForecast.Longterm.scaled_data,
                                  model_path=f'{MODEL_PATH}/{ConsumptionForecast.Longterm.directory}/{country}',
                                  y_data=ConsumptionForecast.Longterm.scaled_y,
                                  country_code=country).create_model(epochs=1000, batch_size=32, train_size=0.97,
                                                                     shuffle=True, patience=60, shuffle_on_train=True)
                print(f' \n Longterm consumption() model for {country} has been retrained \n')

        if longterm:
            long_df = ThunderbirdPredict(type='nn',
                                 raw_data=ConsumptionForecast.Longterm.raw_data,
                                 scaler_name=f'{ConsumptionForecast.Longterm.scaler_name}_{country}.pkl',
                                 model_path=f'{MODEL_PATH}/{ConsumptionForecast.Longterm.directory}/{country}',
                                 y=ConsumptionForecast.y.value,
                                 country_code=country,
                                 backtest=self.backtest).im_predict(mfc_microservice_id=self.mfc_microservice_id,
                                                                          mfc_scenario=self.scenario,
                                                                          mfc_iteration=self.mfc_iteration)
            long_df['mfc_val_1'] = long_df['mfc_val_1'] * -1
            if self.scenario == 5:
                insert_into_table(long_df, 'im', 'im_markets_forecast_calc', primary_key=False,
                        constraint='unique_constraint')
            else:
                self._insert_different_scenarios(td_df=long_df, country=country)
            print(f'Longterm consumption() output prediction for {country} has been done')
        if weekahead:
            week_df = ThunderbirdPredict(type='nn',
                                 raw_data=ConsumptionForecast.Weekahead.raw_data,
                                 scaler_name=f'{ConsumptionForecast.Weekahead.scaler_name}_{country}.pkl',
                                 model_path=f'{MODEL_PATH}/{ConsumptionForecast.Weekahead.directory}/{country}',
                                 y=ConsumptionForecast.y.value,
                                 country_code=country,
                                 backtest=self.backtest).im_predict(mfc_microservice_id=self.mfc_microservice_id,
                                                                          mfc_scenario=self.scenario,
                                                                          mfc_iteration=self.mfc_iteration)
            week_df['mfc_val_1'] = week_df['mfc_val_1'] * -1
            if self.scenario == 5:
                insert_into_table(week_df, 'im', 'im_markets_forecast_calc', primary_key=False,
                        constraint='unique_constraint')
            else:
                self._insert_different_scenarios(td_df=week_df, country=country)
            print(f'WA consumption() output prediction for {country} has been done')
        if two_days_ahead:
            td_df = ThunderbirdPredict(type='nn',
                                 raw_data=ConsumptionForecast.TwoDaysAhead.raw_data,
                                 scaler_name=f'{ConsumptionForecast.TwoDaysAhead.scaler_name}_{country}.pkl',
                                 model_path=f'{MODEL_PATH}/{ConsumptionForecast.TwoDaysAhead.directory}/{country}',
                                 y=ConsumptionForecast.y.value,
                                 country_code=country,
                                 backtest=self.backtest).im_predict(mfc_microservice_id=self.mfc_microservice_id,
                                                                          mfc_scenario=self.scenario,
                                                                          mfc_iteration=self.mfc_iteration)
            td_df['mfc_val_1'] = td_df['mfc_val_1'] * -1
            if self.scenario == 5:
                insert_into_table(td_df, 'im', 'im_markets_forecast_calc', primary_key=False,
                        constraint='unique_constraint')
            else:
                self._insert_different_scenarios(td_df=td_df, country=country)
            print(f'2DA consumption() output prediction for {country} has been done')
        if self.backtest:
            return 'Backtest has been done'
        else:
            return 'Forecast has been done'

    @start_end_microservice_time_with_iteration(4)
    def run(self, mfc_iteration=None):
        LockdownEU()
        if mfc_iteration is None:
            self.mfc_iteration = get_iteration()
        else:
            self.mfc_iteration = mfc_iteration
        if self.countries is None:
            self.countries = ConsumptionForecast.all_countries.value
        for country in self.countries:
            cur.execute(f"SELECT id FROM bi.countries t WHERE t.iso_code = '{country}'")
            id = cur.fetchall()[0][0]
            cur.execute(f"SELECT m_id FROM im.im_market_country "
                        f"WHERE m_commodity = 1 AND m_country = {id}")
            self.mfc_market_id = cur.fetchall()[0][0]
            # logger.info(f"\n \n Backtest 5 for wind onshore generation has not been done yet for {country} \n \n")
            self.backtest = True
            self.scenario = 5
            result = self.consumption(country=country)
            if result == 'Backtest has been done':
                self.backtest = False
                self.scenario = 1
                print('\n \n Backtest for consumption() generation has been done for {country}, time to forecast \n \n')
                self.consumption(country=country)

if __name__ == '__main__':
    Thunder(countries=['SK']).run()
#     df = ThunderbirdPredict(type='nn',
#                        raw_data=ConsumptionForecast.Longterm.raw_data,
#                        scaler_name=f'{ConsumptionForecast.Longterm.scaler_name}_DE.pkl',
#                        model_path=f'{MODEL_PATH}/{ConsumptionForecast.Longterm.directory}/DE',
#                        y=ConsumptionForecast.y.value,
#                        country_code='SK',
#                        backtest=False).im_predict(mfc_microservice_id=30)
#     df = Thunder().merge_domestic_capacity(df=df, country='SK')


    #
    # def fact(self, what=None):
    #     if what == ConsumptionForecast.y.value:
    #         if self.countries == None:
    #             self.countries = ConsumptionForecast.countries.value
    #         for country in self.countries:
    #             fact_data(country_code=country, what=ConsumptionForecast.y.value)
    #     if what == ConsumptionForecast.y.value:
    #         if self.countries == None:
    #             self.countries = ConsumptionForecast.countries.value
    #         for country in self.countries:
    #             fact_data(country_code=country, what=ConsumptionForecast.y.value)