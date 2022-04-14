# CatGAN

Original
paper: [An unsupervised fault diagnosis method for rolling bearing using STFT and generative neural networks Hongfeng Taoa , Peng Wanga , Yiyang Chenb , Vladimir Stojanovicc, Huizhong Yang](https://sci-hub.ru/https://www.sciencedirect.com/science/article/abs/pii/S0016003220302544)

Data was selected
from [Bearing data center datasets](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) and
represents 12KHz sampled vibration accelerometer output for different EDM-damaged fault diameter (0.007, 0.014, 0.021
in) in a different places (ball, inner race, outer race) and different load factor on the rotor.

Each signal have been processed to the time-frequency diagram with subsequent normalization to [0,1].

According to Kotelnikov-Nyquist-Shannon theorem the spectral width of such signal can be up to 6KHz.

High-resolution time-frequency diagrams are above:
![](doc/normal-6k.png?raw=true "Title")

![](doc/ir7-6k.png?raw=true "Title")

Applying narrow segment width for STFT (256 samples) with the subsequent image down-sampling to 64 pixels 
we will have about 0.1 KHz per pixel line. Since the distance between major spectral peaks is greater than 0.1KHz such
down-sampling seems to be correct for the analysis of resulting diagrams.

Raw output after 256-sample STFT (before down-sampling):

![](doc/normal-129.png?raw=true "Title")

![](doc/ir7-129.png?raw=true "Title")

Since we have only one signal for normal data and for each type of faults, we need to cut them into short
overlapping segments to form the train dataset. I'll take 1024 samples per segments with 128samples as a step, so I got
about 1000 segment diagrams for each of 9 fault types and about 2000 diagrams for normal data, total 10408 diagrams.

![](doc/training_set_example.png?raw=true "Title")

The segment step and applied window function properties seems to be appropriate because generated images look different and all spectral characteristics are preserved

For unsupervised classifying damage type was trained categorical generative adversarial network.
Resulting latent manifold is above.

![](doc/unsupervised.png?raw=true "Unsupervised latent manifold")

For comparison, the result of supervised latent manifold:
![](doc/supervised.png?raw=true "Supervised latent manifold")
