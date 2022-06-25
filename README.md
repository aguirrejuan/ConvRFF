# <center> Random Fourier Features in Convolutional Form </center>


$$k(\mathbf{x},\mathbf{x}') = e^{-\frac{(\mathbf{x}-\mathbf{x}')^2}{2 \sigma^2}} \approx z(\mathbf{x})^Tz(\mathbf{x}')$$

$$z(\mathbf{x}): \mathbb{R}^{D} \rightarrow \mathbb{R}^{Q}$$

$$z(\mathbf{x}) = \sqrt{\frac{2}{Q}} [ cos(\mathbf{w}_1^{T}\mathbf{x} + b_1),...,cos(\mathbf{w}_Q^{T}\mathbf{x} + b_Q)]$$

Where $\mathbf{w}_1,...,\mathbf{w}_Q \in \mathbb{R}^D$ is Q i.i.d samples from $p(\mathbf{w}) = \frac{1}{2\pi} \int e^{-j\mathbf{w}'\delta}k(\delta)d\triangle$, and $b_1,...,b_Q \in \mathbb{R}$ is Q samples from $\mathcal{N}(0,2\pi)$

For now
$$p(\mathbf{w}) = (2\pi)^{\frac{2}{Q}}e^{-\frac{\|\mathbf{w}\|_2^2}{2}}$$

**CONVOLUTION**

Properties of translation equivariance and notions of locality.

$\mathbf{F}_l \in \mathbb{R}^{H\times W\times C}$

$\varphi: \mathbb{R}^{H\times W\times C} \rightarrow \mathbb{R}^{H_o\times W_o\times C_o}$

$\sigma \in \mathbb{R}^{+}$ Scale

$$\mathbf{F}_l = \phi(\mathbf{F}_{l-1})= cos(\frac{\mathbf{W}_l}{\sigma}\otimes\mathbf{F}_{l-1}+\mathbf{b}_l)$$


<center>
<figure>
<img src='images/convRFF.png' width="1000"> 
<figcaption></figcaption>
</figure>
</center>



## TODO 
-3. Added parameter layer for averges CAM 
-2. add cross validalidation 
1. Test over more complex dataset (cats and dogs classification)
4. Layer CAM
5. Add compatibily with generator for large datasets. 
7. modify score cam to select input 
8. Inform about the wrong calculation in the notebook (Cristian)
9. Shutdown activation of layer in the middle  
10. https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem
11. Bochner y weiner Jinche próximo viernes 

