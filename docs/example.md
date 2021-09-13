## Examples

### A simple example.

Here is a basic example.

```python
import ConfigModel_MCMC as CM
import networkx as nx

# An example network 
G = nx.gnp_random_graph(n = 100, p = 0.1)

# Specify the graph space and create a new object
allow_loops = False
allow_multi = False
is_vertex_labeled = True
mcmc_object = CM.MCMC(G, allow_loops, allow_multi, is_vertex_labeled)

# Get a new graph (G2) from the Configuration model
G2 = mcmc_object.get_graph()
```

In the above example, `G_2` is sampled from the vertex-labeled simple graph space. The graph space is specified depending on whether the generated network is allowed to have self-loops or not, multi-edges or not, and whether it is stub-labeled or vertex-labeled. `G_2` has the same degree sequence as the example network `G`. Please refer to [[1]](https://epubs.siam.org/doi/pdf/10.1137/16M1087175) for details on how to choose the correct graph space for a given research question.

If no graph space is specified, the simple vertex-labeled graph space will be chosen by default. In the example below, the graph `G_2` is obtained from the simple vertex-labeled graph space.

```python
# An example network 
G = nx.gnp_random_graph(n = 100, p = 0.1)

# Specify the graph space and create a new object
mcmc_object = CM.MCMC(G)

# Get a new graph (G_2) from the Configuration model
G2 = mcmc_object.get_graph()
```


### Sample multiple graphs

Multiple graphs can also be sampled from the Configuration model with/without using a loop.

```python
# An example network 
G = nx.gnp_random_graph(n = 100, p = 0.1)

# Specify the graph space and create a new object
allow_loops = False
allow_multi = False
is_vertex_labeled = True
mcmc_object = CM.MCMC(G, allow_loops, allow_multi, is_vertex_labeled)

# Get 5 graphs from the Configuration model over a loop and print their degree assortativity.
for i in range(5):
    G_new = mcmc_object.get_graph()
    print(round(nx.degree_pearson_correlation_coefficient(G_new),4), end = " ")
print()

# Get 5 more graphs using a single line.
list_of_graphs = mcmc_object.get_graph(count=5)
for each_graph in list_of_graphs:
    print(round(nx.degree_pearson_correlation_coefficient(each_graph),4), end = " ")
print()

```
Output:
```
-0.0564 -0.0177 -0.0583 0.027 0.0778 
-0.0405 -0.0276 -0.0053 0.016 -0.0153
```

In the above code, the first 5 networks are generated over a loop, while the next 5 networks are generated using a single command by specifying ```count=5``` as an argument to the function `get_graph( )`. Both ways are equally efficient on average. The default value of ```count``` is 1. The degree assortativity values of each of the networks generated are printed for reference.

### Sampling Gap heuristics

If the network does not satisfy the conditions under which the automatic selection of the Sampling Gap is possible, the Sampling Gap algorithm will be run. This function might take a while, if the network is large.

```python
# read the Karate Club network
G = nx.karate_club_graph()

# Specify the graph space and create a new object
allow_loops = False
allow_multi = False
is_vertex_labeled = True
mcmc_object = CM.MCMC(G, allow_loops, allow_multi, is_vertex_labeled)

# Obtain 5 graphs from the Configuration model
list_of_graphs = mcmc_object.get_graph(count=5)
```

Output:
```
The network does not satisfy the density criterion for automatic selection of sampling gap.
Running the Sampling Gap Algorithm. This might take a while for large graphs.....
----- Running initial burn-in -----
100%|████████████████████████████         | 78000/78000 [00:33<00:10, 229.00it/s]
----- Initial burn-in complete -----
```

The above code reads the Karate Club network and samples 5 graphs from the vertex-labeled simple graph space. The network does not satisfy the contraints necessary for the automatic selection of the sampling gap, because of its fairly high density. So the Sampling Gap Algorithm is called. A progress bar is displayed during the burn-in period of the MCMC walk. The variable `list_of_graphs` contains the 5 simple vertex-labeled graphs sampled from the Configuration model.

The messages printed in the output can be muted by specifying ```verbose = False``` while creating the MCMC object. The deafult value is ```verbose = True```.

```python
# read the Karate Club network
G = nx.karate_club_graph()

# Specify the graph space and create a new object
allow_loops = False
allow_multi = False
is_vertex_labeled = True
mcmc_object = CM.MCMC(G, allow_loops, allow_multi, is_vertex_labeled, verbose=False)

# Obtain 5 graphs from the Configuration model
list_of_graphs = mcmc_object.get_graph(count=5)
```

### Running the Sampling Gap Algorithm

If you want to run the Sampling Gap Algorithm to obatin a bespoke sampling gap for your graph, you may do so as follows:

```python
# read the Karate Club network
G = nx.karate_club_graph()

# Specify the graph space and create a new object
allow_loops = False
allow_multi = False
is_vertex_labeled = True
mcmc_object = CM.MCMC(G, allow_loops, allow_multi, is_vertex_labeled, verbose=False)

# run the Sampling Gap Algorithm
sampling_gap = mcmc_object.run_sampling_gap_algorithm()
print("Sampling gap obtained = ", sampling_gap)
```

Output:
~~~
Sampling gap obtained = 162
~~~

Note that the sampling gap obtained in each run might vary a bit, although it would be mostly stable around a value. Again, the print statements are muted here by specifying ```verbose = False``` at the time of creating the MCMC object. The print statements of particularly the sampling gap algorithm can be muted using the following code, even when it was not muted while creating the MCMC object.

```python
# read the Karate Club network
G = nx.karate_club_graph()

# Specify the graph space and create a new object
allow_loops = False
allow_multi = False
is_vertex_labeled = True
mcmc_object = MCMC(G, allow_loops, allow_multi, is_vertex_labeled)

# run the Sampling Gap Algorithm
sampling_gap = mcmc_object.run_sampling_gap_algorithm(verbose=False)
print("Sampling gap obtained = ", sampling_gap)
```

Output:
~~~
Sampling gap obtained = 159
~~~


The default significance level of the autocorrelation hypothesis tests = 0.04 and the default number of parallel MCMC chains run for the Sampling Gap Algorithm = 10. However, you can change them by specifying as arguments to the function. For example, we can set the significance level as 10% and run 20 parallel MCMC chains as follows:

```python
# read the Karate Club network
G = nx.karate_club_graph()

# Specify the graph space and create a new object
allow_loops = False
allow_multi = False
is_vertex_labeled = True
mcmc_object = MCMC(G, allow_loops, allow_multi, is_vertex_labeled, verbose=False)

# run the Sampling Gap Algorithm
sampling_gap = mcmc_object.run_sampling_gap_algorithm(alpha = 0.1, D = 20)
print("Sampling gap obtained = ", sampling_gap)
```

Output:
~~~
Sampling gap obtained = 189
~~~

Since both significance level and the number of parallel chains have been increased (from 0.04 to 0.1 and from 10 to 20, respectively), the Sampling Gap will be higher than what was obtained before since the test is more conservative now, and the Sampling Gap Algorithm would take more time to run in this case.


### Customising the sampling gap

You can also specify a custom sampling gap that you want to run the convergence detection test with, using the ```sampling_gap``` function parameter.

```python
# read the Karate Club network
G = nx.karate_club_graph()

# Specify the graph space and create a new object
allow_loops = False
allow_multi = False
is_vertex_labeled = True
mcmc_object = MCMC(G, allow_loops, allow_multi, is_vertex_labeled, verbose=False)

# Specify the sampling gap 
gap = 100 # any user-defined value
list_of_graphs = mcmc_object.get_graph(count=5, sampling_gap = gap)
for each_graph in list_of_graphs:
    print(round(nx.degree_pearson_correlation_coefficient(each_graph),4), end = " ")

print()
print("User-defined sampling gap = ", mcmc_object.spacing)
```
Output:
```
-0.298 -0.2739 -0.3224 -0.3042 -0.2396
User-defined sampling gap =  100
```
