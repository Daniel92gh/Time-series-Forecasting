import torch

from forecasting_library.data.dataset import Dataset
from forecasting_library.utils.logger import logger


class RegressionMapDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)

        # build feature vector
        self._build_feature_vector()
        logger.info(f"Data is ready for train and test for Regression model")


    def _build_feature_vector(self):
        
        '''
        Creates the feature vector for Regression model

        '''    

        # get feature matrix and convert to numy 
        self.x_train = torch.tensor(self._train_x.values).float()
        self.x_dev = torch.tensor(self._dev_x.values).float()
        self.x_test = torch.tensor(self._test_x.values).float()
