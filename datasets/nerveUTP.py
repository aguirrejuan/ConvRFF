from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd


import os
from glob import glob 


import gdown
import zipfile


FILEID = "1GewZspflKFgN7Clut5Xqr3E3CQLSfoYU"


def unzip(file_path,destination_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

def download(destination_path):
    url = f"https://drive.google.com/uc?id={FILEID}"
    output = os.path.join(destination_path,"ImagenesNervios.zip")

    if os.path.exists(output):
        return
    os.makedirs(destination_path,exist_ok=True)
    gdown.download(url, output, quiet=False)
    unzip(output,destination_path)

def preprocessing_mask(mask):
  mask[mask > 0.5] = 255
  mask[mask <= 0.5] = 0
  return mask


def get_data(seed = 1993, batch_size = 32,height,width =(256,256)):


    destination_path = os.path.dirname(__file__)
    destination_path = os.path.join(destination_path,'nerviosUTP')
    download(destination_path)
    destination_path = os.path.join(destination_path,'ImagenesNervios_')

    file_images = glob(os.path.join(destination_path,'*.png'))
    file_images.sort()
    filepath_image = [] # sólo imagenes
    filepath_mask = [] # mascaras
    nerve_name = []
    for filepath in [filepath_ for filepath_ in file_images if 'mask' not in filepath_]:
      mask = filepath[:-4]+'_mask.png'
      if mask in file_images:
        filepath_image.append(filepath)
        filepath_mask.append(mask)

        if 'ciatico' in filepath:
          nerve_name.append('ciatico')
        elif 'cubital' in filepath:
          nerve_name.append('cubital')
        elif 'femoral' in filepath:
          nerve_name.append('femoral')
        elif 'mediano' in filepath:
          nerve_name.append('mediano')

    df = pd.DataFrame({'filepath':filepath_image,'nerve_name':nerve_name,'mask':filepath_mask})
    t = df['nerve_name']
    df_train_images,df_test_images,t_train,_ = train_test_split(df,t, test_size=0.2,stratify = t)
    df_train_images,df_val_images,_,_ = train_test_split(df_train_images,t_train, test_size=0.2,stratify = t_train)

    
    image_datagen = ImageDataGenerator(rotation_range=10,
                                       horizontal_flip =True,
                                       vertical_flip=True,
                                       rescale=1./255)

    image_datagen_mask = ImageDataGenerator(rotation_range=10,
                                       horizontal_flip =True,
                                       vertical_flip=True,
                                       rescale=1./255,
                                       preprocessing_function = preprocessing_mask)

    generator_train_img = image_datagen.flow_from_dataframe(df_train_images,
                                                            x_col = 'filepath',
                                                            batch_size = batch_size,
                                                            class_mode = None,
                                                            directory = None,
                                                            target_size = (height,width),
                                                            seed = seed)

    generator_train_mask = image_datagen_mask.flow_from_dataframe(df_train_images,
                                                             x_col='mask',
                                                             class_mode = None,
                                                             directory = None,
                                                             color_mode="grayscale", 
                                                             batch_size = batch_size,
                                                             target_size = (height,width),
                                                             seed = seed)

    train_gen = zip(generator_train_img,generator_train_mask)


    generator_val_img = image_datagen.flow_from_dataframe(df_val_images,
                                                            x_col='filepath',
                                                            class_mode = None,
                                                            directory = None,
                                                            target_size = (height,width),
                                                            batch_size = batch_size,
                                                            seed = seed)

    generator_val_mask = image_datagen_mask.flow_from_dataframe(df_val_images,
                                                             x_col='mask',
                                                             class_mode = None,
                                                             directory = None,
                                                             color_mode="grayscale",
                                                             target_size = (height,width),
                                                             batch_size = batch_size,
                                                             seed = seed)

    val_gen = zip(generator_val_img,generator_val_mask)


    generator_test_img = image_datagen.flow_from_dataframe(df_test_images,
                                                            x_col='filepath',
                                                            class_mode = None,
                                                            directory = None,
                                                            target_size = (height,width),
                                                            batch_size = batch_size,
                                                            seed = seed)

    generator_test_mask = image_datagen_mask.flow_from_dataframe(df_test_images,
                                                             x_col='mask',
                                                             class_mode = None,
                                                             directory = None,
                                                             color_mode="grayscale",
                                                             target_size = (height,width),
                                                             batch_size = batch_size,
                                                             seed = seed)

    test_gen = zip(generator_test_img,generator_test_mask)




    return train_gen, val_gen, test_gen, len(df_train_images) 




if __name__ == '__main__':
    train_gen, val_gen, test_gen = get_data()
    
