import matplotlib.pyplot as plt 
import numpy as np 

from gcpds.image_segmentation.datasets.segmentation import InfraredThermalFeet
from convRFF.data import get_data

train, val, test = get_data(InfraredThermalFeet, flip_up_down=False, batch_size=1,
                            range_rotate=(-10,10))


for x,y in train:
    x = np.squeeze(x)
    y = np.squeeze(y)

    plt.subplot(1,2,1)
    plt.imshow(x)
    plt.subplot(1,2,2)
    plt.imshow(y)
    plt.show()




