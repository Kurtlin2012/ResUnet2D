import numpy as np
from keras.utils import Sequence

class Dataset(Sequence):
    def __init__(self, df, batch_size):
        self.x = df['X'].values.tolist()
        self.y = df['y'].values.tolist()
        self.bs = batch_size
        
    def __len__(self):
        return len(self.x) // self.bs
    
    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        for i in range(idx*self.bs, (idx+1)*self.bs):
            batch_x.append(np.load(self.x[i]))
            batch_y.append(np.load(self.y[i]))
        return np.stack(batch_x, axis=0), np.stack(batch_y, axis=0)