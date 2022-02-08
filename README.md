# CatGAN

Original
paper: [An unsupervised fault diagnosis method for rolling bearing using STFT and generative neural networks Hongfeng Taoa , Peng Wanga , Yiyang Chenb , Vladimir Stojanovicc, Huizhong Yang](https://sci-hub.ru/https://www.sciencedirect.com/science/article/abs/pii/S0016003220302544)

Data was selected
from [Bearing data center datasets](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) and
represenets 12KHz sampled vibration accelerometer output for different EDM-damaged fault diameter (0.007, 0.014, 0.021
in) in a different places (ball, inner race, outer race) and different load factor on the rotor.

Each signal have been processed to the time-frequency diagram with subsequent normalization to [0,1]

![](doc/Normal.png?raw=true "Title")
![](doc/IR7.png?raw=true "Title")

Since we have only one signal for normal data and for the each type of faults, we need to cut them into short
overlapping pieces to form the train dataset. I'll take 1024 samples per segments with 128samples as a step, so I got
about 1000 segment diagrams for inner race 0.007inch fault and about 2000 diagrams for normal data.

![](doc/training_set_none.png?raw=true "Title")
![](doc/training_set_ir7.png?raw=true "Title")

The segment step and applied window function properties seems to be appropriate because generated images look different and all spectral characteristics are preserved
