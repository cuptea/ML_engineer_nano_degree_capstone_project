import json

from pathlib import Path
from tqdm import tqdm
from exp import Exp

def save_result(file_path, data):
    with open(file_path, 'w') as fp:
        json.dump(data, fp)

input_data_path = Path("/home/ntao/project/data/train.csv")
output_data_path = Path("/home/ntao/project/data/result.csv")
model_dir = "/home/ntao/project/data/models/"

exp = Exp(data_file_name=input_data_path,
          eval_result_file_name=output_data_path,
          model_dir = model_dir,
          n_dim = 10,
          epochs = 100,
          batch_size = 512,
          gpus = [2,3,4,5])
    
exp.run()
    
