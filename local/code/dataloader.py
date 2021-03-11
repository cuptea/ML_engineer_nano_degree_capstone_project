from torch.utils.data import Dataset

class KKbox_set(Dataset):
    def __init__(self, df, transformation=None):
        super(KKbox_set).__init__()
        self.data = df
        self.data = list(self.data.T.to_dict().values())
        self.transform = transformation     
        
    def __getitem__(self,index):
        """return a role assignment in a dictionary form"""
        sample = self.data[index]
        return sample
    
    def __len__(self):
        return len(self.data)