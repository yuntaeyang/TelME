from torch.utils.data import Dataset, DataLoader
import random

 
class iemocap_dataset(Dataset):
    def __init__(self, data):
        self.emoList = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        self.session_dataset = data

    def __len__(self): 
        return len(self.session_dataset)
    
    def __getitem__(self, idx): 
        return self.session_dataset[idx]