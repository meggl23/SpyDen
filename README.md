# SpyDen
***
## Introduction
A code for analyzing structures along and including dendritic stretches.

With SpyDen, you are able to:
- Study a variety of experimental paradigms related to dendritic stretches, both with multiple channels and snapshots.
- Automatically segment dendrites from a variety of experimental paradigms to obtain relevant metrics and statistics.
- Annotate spines (either manually or using an ML approach) to obtain information about the synapses.
- Study the concentration of proteins (as puncta) throughout soma, dendrites, and synapses.

***
## Installing SpyDen

There are two ways to install and run SpyDen.
1. Cloning this repository by running: `git clone`

The first approach will require you to make sure your environment includes all relevant dependencies but allows you easy access to the codebase. This approach is recommended for users who have some knowledge of coding and how to install packages independently.

2. Installing via conda: `conda install spyden`
For those with minimal coding experience, and also in general, we recommend the second approach. This will do all the heavy lifting for you and allow you to get to analyzing faster. Firstly, you need to install miniconda (see link [here](https://docs.conda.io/projects/miniconda/en/latest/)). If you are on Windows, this will provide you with a terminal (anaconda terminal) that allows you to run command line code. On Mac, the terminal is already pre-installed. Then you can run the above mentioned `conda install spyden`. This will create an environment that has all necessary packages and updating is as easy as `conda update spyden`. 

A tutorial for installing SpyDen for both Windows and Mac using both methods can be found [here](www.google.com)

***
## Running SpyDen
### Github installation
If you installed SpyDen by cloning this repository and have made sure all relevant dependencies are also installed. Then you can navigate to the folder where you cloned the repository and run: `python spyden.py`

This will load up the GUI and then you are free to go.

### Conda installation
In your terminal and in the environment where you installed spyden run: `python -m spyden`

This will load up the GUI and then you are free to go.

***
## Tutorials
To find information on how to operate, we suggest that users watch the following videos:
- [Loading data into Spyden](https://www.youtube.com/watch?v=3GOStVqGbA0)
- [Getting started analyzing](https://www.youtube.com/watch?v=dYi8-B6OIv4)
- [Analyzing dendrites](https://www.youtube.com/watch?v=wxRVMRkTVoY)
- [Analyzing synapses](https://www.youtube.com/watch?v=i6YGx5wq2VY)
- [Analyzing puncta](https://www.youtube.com/watch?v=TXSsa4Zr4Ao)
- [Understanding the output](https://www.youtube.com/watch?v=k4r61ijv_ek)

The links to these tutorials can also be found in the *Tutorials* section of SpyDen.

***
## Dependencies
```
matplotlib
numpy
networkx
scikit-image
regex
torch>=2.0
torchvision>=0.15
tifffile
opencv-python
PyQt5
imageio
roifile
```

***
## How to cite
The preprint of SpyDen is currently being prepared.
