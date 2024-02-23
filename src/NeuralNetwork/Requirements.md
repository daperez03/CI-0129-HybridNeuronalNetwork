# Requirements

To use these tools, you need to have Python installed. Additionally, you can install the required dependencies using pip:

```shell
pip install albumentations
pip install matplotlib
pip install nnfs
pip install numpy
pip install Pillow
pip install scipy
pip install scikit-image
pip install tabulate
```

Also, if you want optimized performance and you have a Nvidia GPU, you should install cupy, doing this:

```shell
conda update conda
conda update --all
conda config --add channels conda-forge

conda install numba
conda install cudatoolkit
conda install cupy
```

If you don't want to do this, go to "Operations.py" and change the "cupy" import for "numpy". Cupy module is only used for optimization of vectorial operations. Also, to run the program (if using cupy), do it on the created enviroment that has all these libraries installed.
