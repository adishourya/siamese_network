# Siamese network with triplet loss 

## AT&T Faces dataset 

1. Color: Grey-scale
2. Sample Size: 92x112
3. \#Samples: 400
4. Dataset Size: 4.5 MB (compressed in .tar.z)

The original files are in PGM format, and can conveniently be viewed on UNIX ™ systems using the ‘xv’ program. The size of each image is 92x112 pixels, with 256 grey levels per pixel. The images are organised in 40 directories (one for each subject), which have names of the form sX, where X indicates the subject number (between 1 and 40). In each of these directories, there are ten different images of that subject, which have names of the form Y.pgm, where Y is the image number for that subject (between 1 and 10).

The AT&T face dataset, “(formerly ‘The ORL Database of Faces’), contains a set of face images taken between April 1992 and April 1994 at the lab. The database was used in the context of a face recognition project carried out in collaboration with the Speech, Vision and Robotics Group of the Cambridge University Engineering Department.”

“There are 10 different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement). A preview image of the Database of Faces is available.”

![](/home/anton/Documents/gl/side_missions/siamese_network/faces.png)



## Triplet Loss

**Triplet loss** is a [loss function](https://en.wikipedia.org/wiki/Loss_function) for [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) where a baseline (anchor) input is compared to a positive (truthy) input and a negative (falsy) input. The distance from the baseline (anchor) input to the positive (truthy) input is minimized, and the distance from the baseline (anchor) input to the negative (falsy) input is maximized.[

It is often used for [learning similarity](https://en.wikipedia.org/wiki/Similarity_learning) for the purpose of learning embeddings, like [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) and even [thought vectors](https://en.wikipedia.org/wiki/Thought_vector), and [metric learning](https://en.wikipedia.org/wiki/Metric_learning).

The [loss function](https://en.wikipedia.org/wiki/Loss_function) can be described using a [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) function



This can then be used in a cost function, that is the sum of all losses, which can then be used for minimization of the posed [optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) problem



The indices are for individual input vectors given as a triplet. The triplet is formed by drawing an anchor input, a positive input that describes the same entity as the anchor entity, and a negative input that does not describe the same entity as the anchor entity. These inputs are then run through the network, and the outputs are used in the loss function.

In [computer vision](https://en.wikipedia.org/wiki/Computer_vision) a prevailing belief has been that the triplet loss is inferior to using [surrogate losses](https://en.wikipedia.org/w/index.php?title=Surrogate_loss&action=edit&redlink=1) followed by separate metric learning steps. Alexander Hermans, Lucas Beyer, and Bastian Leibe showed that for models trained from scratch, as well as pretrained models, a special version of triplet loss doing end-to-end deep metric learning outperforms most other published methods as of 2017

![](/home/anton/Documents/gl/side_missions/siamese_network/triplet_loss.jpg)



