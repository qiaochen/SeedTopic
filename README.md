# SeedTopic
SeedTopic: Knowledge-informed topic modelling of spatial transcriptomics for efficient cell typing

[](https://github.com/qiaochen/SeedTopic/blob/main/resources/Fig1_update_logo.pdf)

## Create environment
Using micromamba (change micromamba to conda if you are using conda package management)
```bash
micromamba create -n seededntm python=3.11 -y
micromamba install gxx_linux-64 gcc_linux-64 -c conda-forge -y
micromamba install pyproj -c conda-forge -y
micromamba install -c conda-forge cmake -y
micromamba install -c conda-forge hdf5 -y
micromamba install -c conda-forge pyarrow -y
micromamba install matplotlib-inline -c conda-forge -y
micromamba install ipython -c conda-forge -y
```

Install package
```bash
git clone https://github.com/qiaochen/SeededNTM.git
cd SeededNTM
pip install .
```


### Run topic modelling in commandline
Refer to [demo notebook](https://github.com/qiaochen/SeededNTM/blob/main/demo/demo.ipynb) for input preperation and model training


