import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


from dataloader import KKbox_set
from evaluator import Evaluator
from model import LitMF, LitMLP, LitNCF

class Exp():
    """The class is used to organize model evaluation for content_based 
    recommendation system given the data with meta info of id and role."""
    
    def __init__(self,
                 data_file_name,
                 eval_result_file_name,
                 model_dir,
                 n_dim,
                 epochs,
                 batch_size,
                 gpus):
        
        self.input_file_name = data_file_name
        self.eval_result_file_name = eval_result_file_name
        self.model_dir = model_dir
        self.n_dim = n_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpus = gpus
        self._set_random_seed(42)

    def _set_random_seed(self, seed):
        my_seed = seed
        random.seed(my_seed)
        np.random.seed(my_seed)
    
    def _prepare_data(self, eps=0.01):
        """read data and turn it into the form RecSys model can take as input

        Params:
            file_path: file name 
            eps(float): the value to adjust the values so that NMF can work
        Returns:
            formated(pd.DataFrame): the dataframe can be feed into RecSys models
        """

        data = pd.read_csv(self.input_file_name)
        #debug
        data = data.sample(500000)

        # memory usage by the prepared data
        mem_usage = data.memory_usage(index=True, deep=True).sum()
        print('memory usage by the prepared data is {:.02f} MB'.format(mem_usage/1e6))
     
        return data

    def _my_train_test_split(self):
        """Split the data into Train and Test dataset and prepare it as surprise dataset object
        """
        # This step try to replicate the train test split for collaborative filtering models
        #test_size = 
        train_valid, test = train_test_split(self.data, test_size=0.05)
        
        train_valid = KKbox_set(train_valid)
        test = KKbox_set(test)        

        return train_valid, test

    def _prepare_models(self):
        """All recommendation system algorithms are defined here

        Returns:
            models(dict)
        """
        
        # prepare the model
        mf = LitMF(n_dim=self.n_dim)       
        mlp = LitMLP(n_dim=self.n_dim)
        ncf = LitNCF(n_dim=self.n_dim)

        models = {
            'model_01_MF'  : mf,
            'model_02_MLP' : mlp,
            'model_03_NCP' : ncf,
        }
        return models

    def _save_result(self, file_path, data):
        with open(file_path, 'w') as fp:
            json.dump(data, fp)

    def run(self):
        """Cross validate all models with training dataset and evaluate the performance on the test dataset.
        The eval result will be saved to the eval_result_file_name path."""
        self.data = self._prepare_data()
        train_valid, test =  self._my_train_test_split()
        self.models = self._prepare_models()
        
        lengths = [int(len(train_valid)*0.9), len(train_valid)-int(len(train_valid)*0.9)]       
        
        train, valid = random_split(train_valid, lengths)

        self.eval_result={}
        for (model_name, model) in self.models.items():
            print('\n --- The model under evaluation is {}. ---\n'.format(model_name))
            
            checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                                  dirpath=self.model_dir,
                                                  filename=model_name+'-{epoch:02d}-{valid_loss:.2f}_'+self.input_file_name.name)
            
            earlystopping_callback = EarlyStopping(monitor='valid_loss')
            
            trainer = pl.Trainer(max_epochs=self.epochs, 
                                 logger=False,
                                 callbacks=[earlystopping_callback],
                                 #callbacks=[earlystopping_callback, checkpoint_callback],
                                 gpus=self.gpus,
                                 accelerator='ddp',
                                 auto_lr_find=True)           
            # start training
            trainer.fit(model,
                        train_dataloader=DataLoader(train,batch_size=self.batch_size,num_workers=4),
                        val_dataloaders=DataLoader(valid,batch_size=self.batch_size,num_workers=4))
            
            
            mse=pl.metrics.regression.MeanSquaredError()
            mae=pl.metrics.regression.MeanAbsoluteError()
            
            
            valid_dl = DataLoader(valid,batch_size=512)
            y_val,y_hat_val = [], []
            print("\n--> Evaluating on validation dataset...")
            for batch in tqdm(valid_dl):
                y_val_i,y_hat_val_i,_ = model.predict(batch)
                y_val.append(y_val_i)
                y_hat_val.append(y_hat_val_i)
            y_val = torch.cat(y_val)
            y_hat_val = torch.cat(y_hat_val)
            print(y_val.shape)
            print(y_hat_val.shape)
            print("\n--> Complete evaluation on validation dataset.")
                
            print("\n--> Evaluating on test dataset...")
            test_dl = DataLoader(test,batch_size=512)
            y_test,y_hat_test,identity_test = [], [], []
            for batch in tqdm(test_dl):
                y_test_i,y_hat_test_i,identity_test_i = model.predict(batch)
                y_test.append(y_test_i)
                y_hat_test.append(y_hat_test_i)
                identity_test.append(identity_test_i)
            y_test = torch.cat(y_test)
            y_hat_test = torch.cat(y_hat_test)
            identity_test = torch.cat(identity_test)
            identity_test = [int(i) for i in identity_test]
            print("\n--> Complete evaluation on test dataset.")
            
            self.eval_result[model_name] = dict()
            self.eval_result[model_name]['valid_rmse']=[float(torch.sqrt(mse(y_val,y_hat_val)))]
            self.eval_result[model_name]['valid_mae']=[float(mae(y_val,y_hat_val))]

            self.eval_result[model_name]['test_rmse'] = float(torch.sqrt(mse(y_test,y_hat_test)))
            self.eval_result[model_name]['test_mae'] = float(mae(y_test,y_hat_test))
            
            y_test = [float(i) for i in y_test]
            y_hat_test = [float(i) for i in y_hat_test]

            evaluator = Evaluator(y=y_test,
                                  y_hat=y_hat_test,
                                  identity=identity_test)

            self.eval_result[model_name].update(evaluator.run())
            print("\n--> Complete evaluation on test dataset.")
        self._save_result(self.eval_result_file_name, self.eval_result)
        #return self.eval_result