import tensorflow as tf 
import numpy as np 

class GradientAscent:
  def __init__(self,model,filters=64,img_width=128,img_height=128):
    self.img_width = img_width
    self.img_height = img_height
    self.filters = filters 
    self.model = model 

  def compute_loss(self, image, filter_index):
    activation = self.model(image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)
  
  @tf.function
  def gradient_ascent_step(self,image,filter_index,learning_rate):
      with tf.GradientTape() as tape:
          tape.watch(image)
          loss = self.compute_loss(image,filter_index)
      grads = tape.gradient(loss,image)
      grads = tf.math.l2_normalize(grads)
      image += learning_rate * grads
      return image
  
  def generate_filter_pattern(self,filter_index):
      iterations = 30
      learning_rate = 10.
      image = tf.random.uniform(
          minval =0.4,
          maxval = 0.6,
          shape = (1, self.img_width, self.img_height,3)
      )
      for i in range(iterations):
          image = self.gradient_ascent_step(image,filter_index,learning_rate)
      return image[0].numpy()

  def deprocess_image(self,image):
      image -= image.mean()
      image /= image.std()
      image *= 64 
      image += 124 
      image = np.clip(image,0,255).astype('uint8')
      image = image[25:-25,25:-25,:]
      return image
  
  def get_image(self):
    all_images = []
    for filter_index in range(self.filters):
        #print(f"Processing filter {filter_index}")
        image = self.deprocess_image(self.generate_filter_pattern(filter_index))
        all_images.append(image)

    margin = 5
    n = 8
    cropped_width = self.img_width - 25 * 2
    cropped_height = self.img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))
    for i in range(n):
        for j in range(n):
            image = all_images[i * n + j]
            row_start = (cropped_width + margin) * i
            row_end = (cropped_width + margin) * i + cropped_width
            column_start = (cropped_height + margin) * j   
            column_end = (cropped_height + margin) * j + cropped_height

            stitched_filters[row_start: row_end,
                             column_start: column_end, :] = image

    stitched_filters = stitched_filters.astype('uint8')                         
    #tf.keras.utils.save_img(f"filters_for_layer.png", stitched_filters)
    return stitched_filters
    

if __name__ == "__main__":
    ga = GradientAscent()
    