# List of Experiments 

* [Experiment 1 - Implementation Test RFF vs ConvRFF](https://colab.research.google.com/drive/1K4DQL8AuPwTNQ7MZGzWH4ZF59nfxhSZv?usp=sharing): Comparison between [tf.keras.layers.experimental.RandomFourierFeatures](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/RandomFourierFeatures) and [ConvRFF](https://github.com/aguirrejuan/ConvRFF/blob/master/convRFF/models/convRFF.py) on MNIST dataset,using the result of the dot product of the maps of each one. **The results show that in different output dimensions both accomplish similar dot product results.**

* [Experiment 2 - Under Noise Conditions CNN vs Full ConvRFF](https://colab.research.google.com/drive/15yLFXzMWZG-dgBKVwiHLMXPfruoJpBPz?usp=sharing): Comparison between baseline CNN and ConvRFF under conditions of noise on MNIST dataset. Both models have similar performance, but CNN wins by a little bit. The scale tends to increase more in deeper layers, generally passing from 2.1 to 12 values, which probably is due to the gradient. The laplace kernel dosen't work in any case. **Conclusion: There is no improvement with respect to the noise in this experiment.**

* [Experiment 3 - Gradient Ascent CNN vs ConvRFF](https://colab.research.google.com/drive/1eX-AbhBhkS6q5pUYexartWz8UC4ScLzl?usp=sharing): How is the input that maximizes the output of CNN layer and ConvRFF?. It seems to be more interesting the output of CNN. On the other hand, **the scale parameter of ConvRFF shows that it influences the frequencies perceived by the operation.**

* [Experiment 4 - Influence of Normalization between $[0,\pi]$ before cosine activation ConvRFF](https://colab.research.google.com/drive/1fqC8PRiqHgamyt8AnRvLWRbCVy38YdV1?usp=sharing): Trying to use the range of the input for cosine between $[0,\pi]$ on MNIST dataset. **The experiment shows that this normalization dosnet ippreve the results**.

* [Experiment - 5]() To see...

* [Experiment - 6  Shallow ConvRFF on MNIST datasets](https://colab.research.google.com/drive/1Xoae-FTfxM4K8d2WNCcMH8L91yZGVIlv?usp=sharing): Comparison between shallow networks (One Convolutional layer and one dense) on FASHION and DIGIT MNIST datasets. **ConvRFF perform worse than FCN and CNN if the normalize image is apply as input, but if the image is not normalize ConvRFF gets underfitting with 0.92 acc in train and 0.79 acc on test.** 


