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

We recommend the second approach for those who already have conda installed and have some experience with coding. This will do all the heavy lifting for you and allow you to get to analyzing faster. Firstly, you need to install miniconda (see link [here](https://docs.conda.io/projects/miniconda/en/latest/)). If you are on Windows, this will provide you with a terminal (anaconda terminal) that allows you to run command line code. On Mac, the terminal is already pre-installed. The conda package of spyden can be found here [here](https://anaconda.org/meggl23/spyden/). Then you can run the above mentioned `conda install spyden`. This will create an environment that has all necessary packages and updating is as easy as `conda update spyden`.

3. For those with no coding experience, we also provide the option to download an exe (windows)/dmg (mac) file directly. This will run out of the box without any need for installations and all versions can be found [here](https://gin.g-node.org/CompNeuroNetworks/SpyDenTrainedNetwork/src/master/Executables).

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
- [Loading data into Spyden](https://youtu.be/1o-l9o2W514)
- [Getting started analyzing](https://youtu.be/1DYjQp4MUGA)
- [Analyzing dendrites](https://youtu.be/bU41g8NW8Ts)
- [Analyzing synapses](https://youtu.be/DiqYDdBQRz8)
- [Analyzing puncta](https://youtu.be/fgDD-Ucr3ms)
- [Understanding the output](https://youtu.be/3QC2gGxzXi0)

The links to these tutorials can also be found in the *Tutorials* section of SpyDen.

***
## Image attribution:
App icon was designed by
<a href="https://www.vectorstock.com/royalty-free-vector/stylish-black-and-white-icon-human-brain-vector-13973264">Vector image by VectorStock / freepik</a>
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
superqt
```

***
## Future features
- Vertical distance of spine from dendrite
- Circle of analysis (spine counting) around a given point
- Spine classification into different types
- 3D analysis
  
***
## How to cite
The preprint of SpyDen can be found [here](https://www.biorxiv.org/content/10.1101/2024.06.07.597872v1). 

Eggl*, M. F., Wagle*, S., Filling*, J. P. Chater, T. E., Goda, Y., Tchumatchenko, T.  (2024). SpyDen: Automating molecular and structural analysis across
spines and dendrites. bioRxiv, 2024-06.

BibTeX entry for LaTeX users:

```bibtex
@article{eggl2024spyden,
  title={SpyDen: Automating molecular and structural analysis across spines and dendrites},
  author={Eggl, Maximilian F and Wagle, Surbhit and Filling, Jean P and Chater, Thomas E and Goda, Yukiko and Tchumatchenko, Tatjana},
  journal={bioRxiv},
  pages={2024--06},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
