# User Guide

## Introduction

**ConfigModel_MCMC** is a tool for sampling networks from the Configuration model, given a network and a graph space. This code package builds upon the Double Edge Swap MCMC Graph Sampler by Fosdick et al. [1]. It detects convergence in the Double Edge Swap MCMC and samples networks from the MCMC's stationary distribution, so that the samples are uniform random draws from the Configuration model.

The corresponding paper can be found on the arXiv [here](https://arxiv.org/abs/2105.12120).

[[1]](https://epubs.siam.org/doi/pdf/10.1137/16M1087175) Bailey K. Fosdick, Daniel B. Larremore, Joel Nishimura, Johan Ugander (2018) Configuring Random Graph Models with Fixed Degree Sequences. SIAM Review 60(2):315–355.

### Why use ConfigModel_MCMC?

The random stub-matching algorithm as described by Newman [2] and also implemented by the [configuration_model](https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.generators.degree_seq.configuration_model.html) function of the [networkx](https://pypi.org/project/networkx/) package in Python works best only for the loopy multigraph space where every stub of network has distinct labels. This is because the graph returned by algorithm is a pseudograph, i.e., the graph is allowed to have both self-loops and parallel edges(multi-edges). Often times practitioners remove the self-loops and collapse the multi-edges in the network returned by the function to get a *simple network*, but this modification changes the degree sequence of the network. It also introduces a bias in the network generating process because the high-degree nodes are more likely to have self-loops and multi-edges attached to them than are the low-degree nodes. Therefore, the network generated is a biased sample. The **ConfigModel_MCMC** package lets you sample an unbiased sample from the Configuration model on eight different graph spaces parameterized by self-loops/no self-loops, multi-edges/no multi-edges and stub-labeled/vertex-labeled.

[[2]](https://epubs.siam.org/doi/pdf/10.1137/S003614450342480) M.E.J. Newman (2003), “The structure and function of complex networks”, SIAM REVIEW 45(2):167-256.


## Installing

`pip install ConfigModel_MCMC`

This package has been tested with Python=3.8 and the required packages numpy==1.21.0, networkx==3.1, scipy==1.8.0, numba==0.56.0, arch==5.3.1, python-igraph==0.11.3 and tqdm==4.62.2. These dependencies are automatically installed while installing the `ConfigModel_MCMC` package.

Make sure the latest version of the package has been installed. To check, execute the following command:

`pip show ConfigModel_MCMC`

Details about the package including summary, version, authors, etc., would be displayed. The version number should be **0.2**. If not, try uninstalling and installing the package again, or execute the following command:

`pip install ConfigModel_MCMC==0.2`



## Set-up

The [arch module](https://pypi.org/project/arch/) uses the [OpenBLAS module](https://www.openblas.net/) for estimating the model parameters for the DFGLS test. Since `OpenBLAS` uses multithreading on its own, it is recommended to limit the number of threads before you start Python, especially if running this package on a high-computing cluster.

For example, if you are using Jupyter Notebook, execute the following commands in your terminal before launching Jupyter.

```
$ export MKL_NUM_THREADS=1
$ export OPENBLAS_NUM_THREADS=1
$ jupyter notebook
```

Or if you are running a script named test.py from your terminal, you may execute the following commands.

```
$ export MKL_NUM_THREADS=1
$ export OPENBLAS_NUM_THREADS=1
$ python test.py
```

This will limit the multithreading to a single thread. You can choose other number of threads e.g., 2, 4, etc, depending on the number of availabile CPU cores. On clusters it is usually recommended to limit it to 1 thread per process.

## Notes

The package will not work for weighted networks, directed networks, hypergraphs, or simplical complexes.

## Feedback and bugs

If you find a bug or you have any questions/feedback, please contact upasanad@seas.upenn.edu.

## License

GNU General Public License v3+