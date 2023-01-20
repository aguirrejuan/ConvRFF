import os 
from glob import glob 
import numpy as np 


class DataRead:
    def __init__(self, file_name):
        self.file_name = file_name
        self._files_name = glob(os.path.join(self.file_name,'*'))
        self._keys = {}
        for fp in self._files_name:
            key = os.path.basename(fp).split('.')[0]
            self._keys[key] = fp 
    
    def __len__(self,):
        return len(self._keys)

    def __getitem__(self, key):
        file_name = self._keys[key]
        with open(file_name, 'rb') as f:
            array = np.load(f)
        return array



class DataWrite:
    def __init__(self, file_name, dtype):
        self.file_name = file_name
        self.dtype = dtype
        os.makedirs(self.file_name, exist_ok=True) 

    def __setitem__(self, key, value):
        file_name = os.path.join(self.file_name, f'{key}.npy')
        array = np.array(value, self.dtype)
        with open(file_name, 'wb') as f:
            np.save(f, array)



def mimic_mmap(file_name, dtype, mode):
    if mode == 'r':
        return DataRead(file_name)
    elif mode == 'w+':
        return DataWrite(file_name, dtype)



if __name__ == '__main__':

    DTYPE = np.dtype([('info_instance', 'U50', (2,2)),  
                  ('cam', np.float32, (2,256, 256, 3))
                 ])
    
    file_name = 'prueba.mimicmmap'
    data = mimic_mmap(file_name, dtype=DTYPE, mode='w+')
    for l in range(10):
        data[f'layer_{l}'] = [('h','l'),('k','l')], np.zeros(shape=((2,256, 256, 3)))

    l_data = mimic_mmap(file_name, dtype=DTYPE, mode='r')

    print(l_data['layer_0'])
