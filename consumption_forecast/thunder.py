import ast
import datetime
import os

import pandas as pd
import psycopg2
from decouple import config as envs
from sqlalchemy import create_engine
from eaglegaze_common.common_utils import insert_into_table
from eaglegaze_common.common_utils import start_end_microservice_time
from eaglegaze_common.thunderbird.get_data_and_predict import ThunderbirdPredict
from eaglegaze_common.thunderbird.nn_train_test import ThunderbirdTrain
from eaglegaze_common.thunderbird.scale_the_data import ThunderbirdScale
from eaglegaze_common.thunderbird.thunderattr import ConsumptionForecast
from consumptionNN import ConsumptionNN
import pathlib

path_files = pathlib.Path(__file__).parent.resolve()
os.environ['SCALER_PATH'] = f"{path_files}/scalers/"
os.environ['MODEL_PATH']= f"{path_files}/models/"
MODEL_PATH = os.environ.get('MODEL_PATH')
engine = create_engine(envs('ALCHEMY_CONNECTION', cast=str))
con = engine.raw_connection()
cur = con.cursor()


class Thunder:

    def __init__(self, countries=None, backtest=False):
        self.countries = countries
        self.backtest = backtest
        self.mfc_microservice_id = 4

    def consumption(self, training_day=6, two_days_ahead=True, weekahead=True, longterm=True, country=None, train=True):
        # Lets extract and rescale our data
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
                                 backtest=self.backtest).im_predict(mfc_microservice_id=self.mfc_microservice_id)
            insert_into_table(long_df, 'im', 'im_markets_forecast_calc')
            print(f'Longterm consumption() output prediction for {country} has been done')
        if weekahead:
            week_df = ThunderbirdPredict(type='nn',
                                 raw_data=ConsumptionForecast.Weekahead.raw_data,
                                 scaler_name=f'{ConsumptionForecast.Weekahead.scaler_name}_{country}.pkl',
                                 model_path=f'{MODEL_PATH}/{ConsumptionForecast.Weekahead.directory}/{country}',
                                 y=ConsumptionForecast.y.value,
                                 country_code=country,
                                 backtest=self.backtest).im_predict(mfc_microservice_id=self.mfc_microservice_id)
            insert_into_table(week_df, 'im', 'im_markets_forecast_calc', primary_key=False,
                              constraint='unique_constraint')
            print(f'WA consumption() output prediction for {country} has been done')
        if two_days_ahead:
            td_df = ThunderbirdPredict(type='nn',
                                 raw_data=ConsumptionForecast.TwoDaysAhead.raw_data,
                                 scaler_name=f'{ConsumptionForecast.TwoDaysAhead.scaler_name}_{country}.pkl',
                                 model_path=f'{MODEL_PATH}/{ConsumptionForecast.TwoDaysAhead.directory}/{country}',
                                 y=ConsumptionForecast.y.value,
                                 country_code=country,
                                 backtest=self.backtest).im_predict(mfc_microservice_id=self.mfc_microservice_id)
            insert_into_table(td_df, 'im', 'im_markets_forecast_calc', primary_key=False,
                              constraint='unique_constraint')
            print(f'2DA consumption() output prediction for {country} has been done')

        if self.backtest:
            return 'Backtest has been done'
        else:
            return 'Forecast has been done'

    @start_end_microservice_time(4)
    def run(self):
        if self.countries is None:
            self.countries = ConsumptionForecast.all_countries.value
        for country in self.countries:
            cur.execute(f"SELECT id FROM bi.countries t WHERE t.iso_code = '{country}'")
            id = cur.fetchall()[0][0]
            cur.execute(f"SELECT m_id FROM im.im_market_country "
                        f"WHERE m_commodity = 1 AND m_country = {id}")
            mfc_market_id = cur.fetchall()[0][0]
            cur.execute(f'SELECT * FROM im.im_markets_forecast_calc '
                        f'WHERE mfc_microservice_id = {self.mfc_microservice_id} '
                        f'AND mfc_scenario = 4 '
                        f'AND mfc_market_id = {mfc_market_id}')
            df = pd.DataFrame(cur.fetchall())
            if len(df) > 0:
                self.backtest = False
            else:
                print(f"\n \n Backtest for consumption() generation has not been done yet for {country} \n \n")
                self.backtest = True
            result = self.consumption(country=country)
            if result == 'Backtest has been done':
                print('\n \n Backtest for consumption() generation has been done for {country}, time to forecast \n \n')
                self.backtest = False
                self.consumption(country=country, train=False)

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