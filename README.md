
This repository contains the implementations in Keras of various methods to understand the prediction by a Convolutional Neural Networks. Implemented methods are:

* Vanila gradient [https://arxiv.org/abs/1312.6034]
* Guided backprop [https://arxiv.org/abs/1412.6806]
* Integrated gradient [https://arxiv.org/abs/1703.01365]
* Visual backprop [https://arxiv.org/abs/1611.05418]

Each of them is accompanied with the corresponding smoothgrad version (), which improves on any baseline method by adding random noise.

Courtesy of https://github.com/tensorflow/saliency, https://github.com/mbojarski/VisualBackProp.

# Examples

* Doberman

<img width="700" src="images/doberman_viz.png">

* Pug and Cat

<img width="700" src="images/cat_dog_viz.png">


# Usage

cd deep-viz-keras

```
from guided_backprop import GuidedBackprop
from utils import *

guided_bprop = GuidedBackprop(vgg16_model)
image = load_image(/path/to/image)
mask = guided_bprop.get_mask(image)               # compute the gradients
show_image(mask)                                  # display the grayscaled mask
```

The examples.ipynb contains the demos of all implemented methods using the built-in VGG16 model of Keras.
