import torch 
import numpy as np

from forecasting_library.data.dataset import Dataset
from forecasting_library.utils.utils import logger


class LSTMMapDataset(Dataset):
    def __init__(self, data_dir, config):
        super().__init__(data_dir)

        # build feature vector
        self._build_feature_vector(config)
        logger.info(f"Data is ready for train and test for LSTM model")


    def _build_feature_vector(self, config):
        
        '''
        Creates the feature vector for LSTM model

        '''

        # get feature matrix and convert to torch tesnor 
        self._x_train = torch.tensor(self._train_x.values).float()
        self._x_dev = torch.tensor(self._dev_x.values).float()
        self._x_test = torch.tensor(self._test_x.values).float()

        # generate feature vector for LSTM, assume LSTM window_size = 24
        window = config.lstm.window_size
        self.x_train = self._generate_timestep_features(self._x_train, window)
        self.x_dev = self._generate_timestep_features(self._x_dev, window)
        self.x_test = self._generate_timestep_features(self._x_test, window)

        # fix y label range correction : window size of 24 hours
        self.y_train = self.y_train[window:]
        self.y_dev = self.y_dev[window:]
        self.y_test = self.y_test[window:]  

        # correct index for lstm by window size
        self._index = self._index[window:]


    @staticmethod
    def _generate_timestep_features(x, window):
        
        '''
        Creates time steps for LSTM by winodw size
        :param x  input feature matrix 
        :param window  window size 

        '''

        x_array = []
        for i in range(len(x) - window):
            row = [a for a in x[i: i + window]]
            x_array.append(row)
        return torch.tensor(np.array(x_array))
