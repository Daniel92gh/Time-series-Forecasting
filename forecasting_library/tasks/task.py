import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from forecasting_library.data.lstm import LSTMMapDataset
from forecasting_library.data.regression import RegressionMapDataset
from forecasting_library.models.lstm import LSTMModel
from forecasting_library.models.regression import LinearRegressionModel
from forecasting_library.utils.utils import *
from forecasting_library.utils.logger import logger
from forecasting_library.utils.early_stopping import EarlyStopping



class TASK:
    def __init__(self, model_name, config_dir, data_dir, work_dir, model_dir, task_type):

        '''
        Abstract class for runing both LTSM and Regression model tarining and testing 
        
        '''

        # general settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        logger.info(f"Using device: {self.device}")
        self._model_name = model_name
        self._config = dict2object(load_config(config_dir))
        self._train = True if task_type == "train" else False
        self._num_epochs = self._config.general.num_epochs

        # work dir, model_dir
        self._work_dir = os.path.join(work_dir)
        self._model_dir = os.path.join(model_dir)

        # check and create common directories
        self._check_create_dirs()

        # set datasets 
        if self._model_name == 'lstm':
            dataset = LSTMMapDataset(data_dir, self._config)
        elif self._model_name == 'regression':
            dataset = RegressionMapDataset(data_dir) 
        if self._train:
            self._dataset_train = TensorDataset(dataset.x_train, dataset.y_train)         
            self._dataset_dev = TensorDataset(dataset.x_dev, dataset.y_dev)
        else:
            self._dataset_test = TensorDataset(dataset.x_test, dataset.y_test)

        # set dataloaders
        if self._train:
            # self._dataloader_train = DataLoader(self._dataset_train, batch_size=self._config.general.batch_size, shuffle=True)
            self._dataloader_train = DataLoader(self._dataset_train, batch_size=self._config.general.batch_size)
            self._dataloader_dev = DataLoader(self._dataset_dev, batch_size=self._config.general.batch_size)
        else:
            self._dataloader_test = DataLoader(self._dataset_test, batch_size=self._config.general.batch_size)

        # set datetime index and target scaler
        self.index = dataset._index
        self._scaler_y = dataset._scaler_y

        # configure model 
        if self._model_name == 'lstm':
            self._model = LSTMModel(self._config)
        elif self._model_name == 'regression':
            self._model = LinearRegressionModel(self._config) 
        logger.info("Model is configured")    

        # load trained model
        if not self._train:
            model_path = os.path.join(self._model_dir, 'model_best_ckpt.pt')
            check_file(model_path)
            self._model.load_state_dict(torch.load(model_path))
            logger.info("Model Loaded !")

        # model summary    
        logger.info(self._model)    

        # set loss function 
        self._loss_function = nn.MSELoss()

        # set optimizer
        if self._train:
            if self._model_name == 'lstm':
                self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.general.lr)
            elif self._model_name == 'regression':
                self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._config.general.lr, momentum=0.9)
        
        # Using early stopping in training
        if self._train:
            self._early_stopping = EarlyStopping(self._config.general.early_stopping_patience,
                                                tolerance=self._config.general.early_stopping_tolerance)


    def _check_create_dirs(self): 

        '''
        Check and creates the required directories
        param: work_dir 
        param: model_dir

        '''

        if self._train:
            if os.path.exists(self._model_dir):
                shutil.rmtree(self._model_dir)
                logger.info("The previous model dir is removed")
            os.makedirs(self._model_dir) 
            logger.info(f"Model dir is created at :{self._model_dir}") 
        else:
            if os.path.exists(self._model_dir):
                logger.info("Model dir exits")

        if self._train:
            if os.path.exists(self._work_dir):
                shutil.rmtree(self._work_dir)
                logger.info("The previous work dir is removed")
            os.makedirs(self._work_dir) 
            logger.info(f"Work directory is created at :{self._work_dir}") 
        else:
            if os.path.exists(self._work_dir):
                logger.info("Wrok dir exits")                    

        # TODO: force to re-create model dir and work_dir  


    def _save_checkpoint(self, epoch):

        '''
        Saves the model checkpoint.

        '''

        model_ckpt_path = os.path.join(self._work_dir, f"model_ckpt_epoch_{epoch}.pt")
        best_model_path = os.path.join(self._model_dir, "model_best_ckpt.pt")
        torch.save(self._model.state_dict(), model_ckpt_path)
        logger.info(f"Saving model checkpoint at epoch {epoch} in {model_ckpt_path}.")
        torch.save(self._model.state_dict(), best_model_path) 


    def run_train_loop(self):

        '''
        Training loop for both tasks

        '''

        logger.info("---------------------------------------------------\n")
        logger.info(f"Number of batches in the eval loop: {len(self._dataloader_train)}")

        self._best_validation_loss = 1000.0

        for epoch in range(self._num_epochs):

            # set model into train mode 
            self._model.train()

            training_loss = 0.0
            epoch_loss = 0.0

            logger.info(f"Training epoch {epoch}:")

            for step, (batch_x, batch_y) in enumerate(self._dataloader_train, start=1):
                
                # initialize optimizer
                self._optimizer.zero_grad(set_to_none=True)                
                
                # forward pass
                predictions = self._model(batch_x)

                # compute loss
                loss = self._loss_function(predictions, batch_y) 

                # update training loss
                training_loss += loss.detach()
                epoch_loss += loss.detach()

                # backward pass 
                loss.backward()

                # # gradient clipping 
                # torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)

                # update weights
                self._optimizer.step()

                # log trainig loss 
                if step % 50 == 0:
                    logger.info(f"STEP : {step}")
                    logger.info(f"TRAINING LOSS : {training_loss/50}; {epoch_loss/step} in epoch")
                    training_loss = 0.0

            self._eval_loop(epoch, test=False, dataloader=self._dataloader_dev)
            if self._early_stopping.early_stop:
                logger.info(f"Early stopping at epoch : {epoch}")
                break       


    def _eval_loop(self, epoch=1, test=False, dataloader=None):

        '''
        Evaluation loop for dev and test set
        
        '''

        # set model to eval mode
        self._model.eval()

        if not self._train:
            self._true = []
            self._pred = []

        with torch.no_grad():

            validation_loss = 0.0 
            ape = 0.0

            logger.info(f"Number of batches in the eval loop: {len(dataloader)}")
            for step, (batch_x, batch_y) in enumerate(dataloader, start=1):
                
                # predict
                predictions = self._model(batch_x)
                loss = self._loss_function(predictions, batch_y)
                validation_loss += loss.item()

                # save results for test
                if not self._train:
                    self._true.extend(batch_y.tolist())
                    self._pred.extend(predictions.tolist())

                # compute average error
                ape_batch = torch.sum(torch.abs((predictions - batch_y) / batch_y) * 100)
                ape += ape_batch

                if not test:
                    if step % 10 == 0:
                        logger.info(f"VALIDATION STEP: {step}")


            # update validation loss and MAPE 
            validation_loss /= len(dataloader) 
            mape = ape / len(dataloader)  

            # results for eval or test 
            res = 'Test' if test else 'Validation'

            logger.info(f"{res} results")
            #logger.info(f"MAPE is {mape} for epoch {epoch} ")
            logger.info(f"Validation loss is {validation_loss} for epoch {epoch} \n")
            
            # save model
            if not test:
                if validation_loss < self._best_validation_loss:
                    self._save_checkpoint(epoch)
                    self._best_validation_loss = validation_loss

                # check early stopping and update validation loss
                self._early_stopping(validation_loss)   

    
    def run_test_loop(self):

        '''
        Runnig test loop
        
        '''

        self._eval_loop(test=True, dataloader=self._dataloader_test)

        # prediction and target re-scaling
        y_pred = self._scaler_y.inverse_transform(pd.DataFrame(self._pred, columns=['CAISO_system_load'], index=self.index))
        y_true = self._scaler_y.inverse_transform(pd.DataFrame(self._true, columns=['CAISO_system_load'], index=self.index))

        # calculate MAPE for Test set
        mape = self._calculate_mape(y_true, y_pred)
        logger.info(f"MAPE for Test set is: {mape}")

        # compute MAPR for 7-Day Forecasting 
        mape = self._calculate_mape(y_true[-7*24:], y_pred[-7*24:])
        logger.info(f"MAPE for 7-Day forecasting is {mape}")

        # plot figures for all test set
        plt.figure(figsize=(10, 5))
        plt.plot(y_pred[-7*24:], label='Predicted') # y_pred[-7*24:]
        plt.plot(y_true[-7*24:], label='True')  # y_true[-7*24:]
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title('LSTM 7-Day Load Forecasting')
        plt.legend()

        # save figures
        out_path = os.path.join(self._work_dir, f"{self._model_name}.png")
        plt.savefig(out_path)
        plt.show()

        # last day prediction 
        last_day = y_pred[-24:]
        logger.info(f"Last day predictions are {last_day}")


    def _calculate_mape(self, y_true, y_pred):
        # avoid zero division
        y_true = np.where(y_true == 0, 1e-8, y_true)
        # calclute mape
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape           

