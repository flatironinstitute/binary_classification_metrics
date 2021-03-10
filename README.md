[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/flatironinstitute/binary_classification_metrics/main)

# binary_classification_metrics

This repository includes:

- Jupyter widget implementations for exporing the behavior of binary classification metrics.
- Implmentations of various binary classification metrics and supporting combinatorics logic.
- A collection of Jupyter notebooks to support a talk about binary classificaiton metrics.

Please run the notebooks using the binder image link at the top of this README.

The Jupyter notebooks for the talk are in the `./notebooks/presentation` folder.  Start with `0_Outline`.

# Local install

If you want to use this presentation and dependant software on an ongoing basis you can install a local copy
on your computer.

You will need an environment with 
<a href="https://ipywidgets.readthedocs.io/en/latest/user_install.html">
Python 3 and Jupyter installed and Jupyter widgets enabled for the repository to work properly.
</a>

To install and run the repository and notebooks on your local machine proceed as follows.

## Install dependancies

Some of the dependancies should be installed from `github` which apparently doesn't always work automically with
the configuration files here (please tell me how to fix them if you know how).  Install those dependancies separately as
follows:

```bash
% pip install https://github.com/AaronWatters/jp_doodle/zipball/master
% pip install https://github.com/AaronWatters/feedWebGL2/zipball/master
```

## Clone the git repository and install the Python module in development mode

First clone the repository in an appropriate folder:

```bash
% git clone https://github.com/flatironinstitute/binary_classification_metrics.git
```

Then change directory into the folder and install the module in development mode

```bash
% cd binary_classification
% pip install -e .
```

To view the notebooks, launch a Jupyter server starting in the
`binary_classification` folder or above it.

```bash
% jupyter notebook
```

Then navigate to `./notebooks/presentation`.





