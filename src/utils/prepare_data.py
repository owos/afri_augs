import torch
import pandas as pd 

class MfandNLLBDatatset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep='\t')
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        xx, yy = [], []
        item = self.data.iloc[idx]
        xx.append(item[self.src_lang].lower())
        yy.append(item[self.tgt_lang].lower())
        return xx, yy, lang1, lang2

        return 
        