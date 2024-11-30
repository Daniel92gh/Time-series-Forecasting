import os
import pandas as pd
import numpy as np
import torch
from dateutil import parser
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler

from forecasting_library.utils.utils import *
from forecasting_library.utils.logger import logger


class Dataset(ABC):
    def __init__(self, data_dir):

        '''
        Reading time seris data
        :param data path to data directory
        :param config task config parameters
        
        '''

        # check data directory and find the csv file
        check_path(data_dir)
        file_name = find_csv(data_dir)

        # read data
        self._df = pd.read_csv(os.path.join(data_dir, file_name))
        logger.info(f"Loaded data from: {data_dir}")
        
        # convert time step format and check if it's a Datetime series 
        self._df['interval_start_time'] = self._df['interval_start_time'].apply(lambda x: parser.isoparse(x).strftime('%Y-%m-%d %H:%M:%S')) 
        self._df['interval_start_time'] = pd.to_datetime(self._df['interval_start_time'])
        self._df.set_index('interval_start_time', inplace=True)

        assert isinstance(self._df.index, pd.DatetimeIndex)
        assert set(self._df.columns) == {   
            "CAISO_system_load",
            "temp_forecast_dayahead_bakersfield",
            "temp_forecast_dayahead_los_angeles",
            "temp_forecast_dayahead_san_francisco",
            "dewpoint_forecast_dayahead_bakersfield",
            "dewpoint_forecast_dayahead_los_angeles",
            "dewpoint_forecast_dayahead_san_francisco" 
        }
        logger.info(f"Data is consistent !!")


        # TODO: check and fill non or missing values => mean value over the missing week
        if self._df.isna().any().any():
            self._df['week'] = self._df.index.to_period('W')
            for col in self._df.columns:
                if col != 'week':
                    self._df[col] = self._df.groupby('week')[col].transform(lambda x: x.fillna(x.mean()))
            self._df.drop(columns='week', inplace=True)
            logger.info("NaN values are filled with mean weekly value")

        # feature exctraction
        ####1 previous load values 
        self._df['24h_ago'] = self._df['CAISO_system_load'].shift(24)
        self._df['25h_ago'] = self._df['CAISO_system_load'].shift(25)
        self._df['26h_ago'] = self._df['CAISO_system_load'].shift(26)
        self._df['48h_ago'] = self._df['CAISO_system_load'].shift(48)
        self._df['49h_ago'] = self._df['CAISO_system_load'].shift(49)
        self._df['50h_ago'] = self._df['CAISO_system_load'].shift(50)
        self._df['72h_ago'] = self._df['CAISO_system_load'].shift(72)
        self._df['73h_ago'] = self._df['CAISO_system_load'].shift(73)
        self._df['96h_ago'] = self._df['CAISO_system_load'].shift(96)
        self._df['97h_ago'] = self._df['CAISO_system_load'].shift(97)
        self._df['120h_ago'] = self._df['CAISO_system_load'].shift(120)
        self._df['121h_ago'] = self._df['CAISO_system_load'].shift(121)  
        self._df['144h_ago'] = self._df['CAISO_system_load'].shift(144)
        self._df['145h_ago'] = self._df['CAISO_system_load'].shift(145)  
        self._df['168h_ago'] = self._df['CAISO_system_load'].shift(168)
        self._df['169h_ago'] = self._df['CAISO_system_load'].shift(169)  

        ####2 extract additional features: hour, day of month, month, weekend, day of week, season, year
        # hour
        self._df['hour'] = self._df.index.hour 
        # peak hour
        self._df['peak_hour'] = ((self._df['hour'] >= 7) & (self._df['hour'] <= 22)).astype(int)
        # day
        self._df['day'] = self._df.index.day
        # month
        self._df['month'] = self._df.index.month 
        # season
        conditions = [  self._df['month'].isin([12, 1, 2]),     # winter
                        self._df['month'].isin([3, 4, 5]),      # spring
                        self._df['month'].isin([6, 7, 8]),      # summer
                        self._df['month'].isin([9, 10, 11])  ]  # fall
        season_values = [0, 1, 2, 3] # season values
        self._df['season'] = np.select(conditions, season_values)
        # week
        self._df['day_of_week'] = self._df.index.dayofweek # 0: Monday, 6:Friday
        # weekend
        self._df['is_weekend'] = (self._df['day_of_week'] >= 5).astype(int)
        # year
        self._df['year'] = self._df.index.year
        ##TODO maybe the additional 10 hours of Day D could be also used in some ways.

        # data consistency
        self._df_cleaned = self._df[192:] # remove first 8 days

        # split into feature and target 
        self._df_x = self._df_cleaned.drop(columns=['CAISO_system_load'])
        self._df_y = self._df_cleaned['CAISO_system_load']

        # scale data for learnig convergence
        self._scaler_x = MinMaxScaler()
        self._x = pd.DataFrame(self._scaler_x.fit_transform(self._df_x), columns=self._df_x.columns, index=self._df_x.index)
        
        self._scaler_y = MinMaxScaler()
        self._y = pd.DataFrame(self._scaler_y.fit_transform(self._df_y.values.reshape(-1, 1)), columns=['CAISO_system_load'], index=self._df_y.index)

        # find index for train, dev, test sets 
        train_size = int(len(self._df_cleaned) * 70 /100)
        dev_size = int(len(self._df_cleaned) * 15 /100)

        # split into train, dev, test sets
        self._train_x = self._x.iloc[:train_size]
        self._train_y = self._y.iloc[:train_size]
        logger.info(f"Training data size:{len(self._train_x)}") 

        self._dev_x = self._x.iloc[train_size: train_size + dev_size]
        self._dev_y = self._y.iloc[train_size: train_size + dev_size]
        logger.info(f"Validation data size:{len(self._dev_x)}")

        self._test_x = self._x.iloc[train_size + dev_size :]
        self._test_y = self._y.iloc[train_size + dev_size :]
        logger.info(f"Test data size:{len(self._test_x)}")

        # store index for testing
        self._index = self._df_cleaned.index[train_size + dev_size :]

        # create target vector and convert to torch tensor 
        self.y_train = torch.tensor(self._train_y['CAISO_system_load'].values).float()
        self.y_dev = torch.tensor(self._dev_y['CAISO_system_load'].values).float()
        self.y_test = torch.tensor(self._test_y['CAISO_system_load'].values).float()

        
    @abstractmethod
    def _build_feature_vector(self):
        
        '''
        Creates the feature vector for Regression model

        '''    

        pass

