"""
Created on Tue Dec 29 2020
Edited on Thu June 2 2021

@author: Upasana Dutta
Version: 0.0.7
"""

import networkx as nx
from scipy import stats
import copy
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import warnings
import math
import numba as nb
from arch.unitroot import DFGLS
import igraph as ig
import time
from collections import defaultdict

jit =  nb.jit
        
__docformat__ = 'reStructuredText'


#If numba can't load, the following do-nothing decorator will apply
def __no_jit__(*dargs,**dkwargs):
    def decorate(func):
        def call(*args,**kwargs):
            return func(*args,**kwargs)
        return call
    return decorate

# This function is adapted with little modification from Joel Nishimura's code @https://github.com/joelnish/double-edge-swap-mcmc
@jit(nopython=True,nogil=True)
def MCMC_step_vertex(A,edge_list,swaps,rejected,allow_loops,allow_multi):
    '''
    
    Performs a vertex-labeled double edge swap.
    
    | Args:
    |     A (nxn numpy array): The adjacency matrix. Will be changed inplace.
    |     edge_list (nx2 numpy array): List of edges in A. Node names should be 
            the integers 0 to n-1. Will be changed inplace. Edges must appear
            only once.
    |     swaps (length 4 numpy array): Changed inplace, will contain the four
            nodes swapped if a swap is accepted.
    |     allow_loops (bool): True only if loops allowed in the graph space.
    |     allow_multi (bool): True only if multiedges are allowed in the graph space.
    
    | Returns:
    |     bool: True if swap is accepted, False if current graph is resampled. 
    
    Notes
    -----
    This method currently requires a full adjacency matrix. Adjusting this 
    to work a sparse adjacency matrix simply requires removing the '@jit'
    decorator.  This method supports loopy graphs, but depending on the degree
    sequence, it may not be able to sample from all loopy graphs.
            
    '''
    # Choose two edges uniformly at random
    m = len(edge_list)
    p1 = np.random.randint(m)
    p2 = np.random.randint(m-1)
    if p1 == p2: # Prevents picking the same edge twice
        p2 = m-1
        
    u,v = edge_list[p1]        
    if np.random.rand()<0.5: #Pick either swap orientation 50% at random
        x,y = edge_list[p2]        
    else:
        y,x = edge_list[p2]
        
    # Note: tracking edge weights is the sole reason we require the adj matrix.
    # Numba doesn't allow sparse or dict objs. If you don't want to use numba
    # simply insert your favorite hash map (e.g. G[u][v] for nx multigraph G).
    w_uv = A[u,v]
    w_xy = A[x,y]
    w_ux = A[u,x]
    w_vy = A[v,y]

    # If multiedges are not allowed, resample if swap would replicate an edge
    if not allow_multi and ( w_ux>=1 or w_vy>=1 ):
        rejected[0] = 1
        return False
    
    num_loops = 0
    unique_nodes = 4
    # Here we count the number of self-loops and unique nodes in our proposed 2 
    # edges. Using the convention where self-loops contribute 2 to the adj matrix
    # requies dividing entries on the diagonal by 2 to get the number of 
    # self-loops.   
    if u == v:        
        num_loops += 1   
        unique_nodes += -1
        w_uv = w_uv/2
    if x == y:
        num_loops += 1
        unique_nodes += -1
        w_xy = w_xy/2
    if u == x:
        unique_nodes += -1
        w_ux = w_ux/2
    if v == y:
        unique_nodes += -1
        w_vy = w_vy/2
    if u == y:
        unique_nodes += -1
    if v == x:
        unique_nodes += -1
    
    if unique_nodes == -2: # If u=v=x=y then unique_nodes currently is -2
        rejected[0] = 1
        return False
    if unique_nodes == 1: # Correcting u=v=x~=y or similar rotation
        unique_nodes = 2


    # We now run through the different possible edge swap cases
    if unique_nodes == 2:
        if num_loops == 2:
            if allow_multi: # Swapping two self-loops creates would create a multiedge
                swapsTo = 2*w_uv*w_xy
                swapsFrom = ((w_ux+2)*(w_vy+1))/2
            else:
                rejected[0] = 1
                return False
        elif num_loops == 1:
            rejected[0] = 1
            return False # The only swap on two nodes with 1 self-loop doesn't change graph
        else: # No_loops on 2 nodes
            if allow_loops:
                swapsTo = (w_uv*(w_xy-1))/2 
                swapsFrom = 2*(w_ux+1)*(w_vy+1) 
            else: # Only change would make 2 self-loops
                rejected[0] = 1
                return False
    elif unique_nodes == 3:
        if num_loops == 0:
            if allow_loops: # Swapping adjacent edges creates a self-loop
                swapsTo = w_uv*w_xy
                swapsFrom = 2*(w_ux+1)*(w_vy+1)
            else: # Only change would make a self-loop
                rejected[0] = 1
                return False
        else: # Num_loops==1
            swapsTo = 2*w_uv*w_xy
            swapsFrom = (w_ux+1)*(w_vy+1)
    else: # Unique_nodes ==4
        swapsTo = w_uv*w_xy
        swapsFrom = (w_ux+1)*(w_vy+1)
    
    # Based upon the above cases we calculate the minimum probability of the 
    # swap and the reverse swap.
    probTo = np.float(swapsTo)
    probFrom = np.float(swapsFrom)
    
    if probFrom/probTo < 1.0:
        P = probFrom/probTo
    else:
        P = 1.0
    

    # If we proceed with the swap we update A, swaps and edge_list
    if np.random.rand() < P:
        swaps[0] = u # Numba currently is having trouble with slicing
        swaps[1] = v
        swaps[2] = x
        swaps[3] = y
        
        A[u,v] += -1
        A[v,u] += -1
        A[x,y] += -1
        A[y,x] += -1
        
        A[u,x] += 1
        A[x,u] += 1
        A[v,y] += 1
        A[y,v] += 1
        
        edge_list[p1] = u,x
        edge_list[p2] = v,y
        return True
    else:
        rejected[0] = 1
        return False
            
# This function is adapted with little modification from Joel Nishimura's code @https://github.com/joelnish/double-edge-swap-mcmc
@jit(nopython=True,nogil=True)
def MCMC_step_stub(A,edge_list,swaps,rejected,allow_loops,allow_multi):
    '''
    
    Performs a stub-labeled double edge swap.
    
    | Args:
    |     A (nxn numpy array): The adjacency matrix. Will be changed inplace.
    |     edge_list (nx2 numpy array): List of edges in A. Node names should be 
            the integers 0 to n-1. Will be changed inplace. Edges must appear
            only once.
    |     swaps (length 4 numpy array): Changed inplace, will contain the four
            nodes swapped if a swap is accepted.
    |     allow_loops (bool): True only if loops allowed in the graph space.
    |     allow_multi (bool): True only if multiedges are allowed in the graph space.
    
    | Returns:
    |     bool: True if swap is accepted, False if current graph is resampled. 
    
    Notes
    -----
    This method currently requires a full adjacency matrix. Adjusting this 
    to work a sparse adjacency matrix simply requires removing the '@nb.jit'
    decorator. This method supports loopy graphs, but depending on the degree
    sequence, it may not be able to sample from all loopy graphs.
        
    '''
    # Choose two edges uniformly at random
    m= len(edge_list)
    p1 = np.random.randint(m)
    p2 = np.random.randint(m-1)
    if p1 == p2: # Prevents picking the same edge twice
        p2 = m-1
        
    u,v = edge_list[p1]        
    if np.random.rand()<0.5: #Pick either swap orientation 50% at random
        x,y = edge_list[p2]        
    else:
        y,x = edge_list[p2]

    # Note: tracking edge weights is the sole reason we require the adj matrix.
    # Numba doesn't allow sparse or dict objs. If you don't want to use numba
    # simply insert your favorite hash map (e.g. G[u][v] for nx multigraph G).
    w_ux = A[u,x]
    w_vy = A[v,y]

    # If multiedges are not allowed, resample if swap would replicate an edge
    if not allow_multi:
        if ( w_ux>=1 or w_vy>=1 ):
            rejected[0] = 1
            return False
            
        if u == v and x == y:
            rejected[0] = 1
            return False
    
    #If loops are not allowed then only swaps on 4 distinct nodes are possible
    if not allow_loops:
        if u == x or u == y or v == x or v == y:
            rejected[0] = 1
            return False
    
   
    swaps[0] = u # Numba currently is having trouble with slicing
    swaps[1] = v
    swaps[2] = x
    swaps[3] = y
    
    A[u,v] += -1
    A[v,u] += -1
    A[x,y] += -1
    A[y,x] += -1
    
    A[u,x] += 1
    A[x,u] += 1
    A[v,y] += 1
    A[y,v] += 1   

    edge_list[p1] = u,x
    edge_list[p2] = v,y
    
    return True


def get_empty_nG(allow_loops, allow_multi):
    if allow_loops and allow_multi:
        G = nx.MultiGraph()
    elif allow_loops and not allow_multi:
        G = nx.Graph()
    elif not allow_multi and not allow_loops:
        G = nx.Graph()
    else:
        G = nx.MultiGraph()
    return G

def get_igraph_G(n, edgeList):
    iG = ig.Graph(directed=False)
    iG.add_vertices(n)
    iG.add_edges(edgeList)
    return iG

def get_networkx_G(n, edgeList, allow_loops, allow_multi):
    nG = get_empty_nG(allow_loops, allow_multi)
    for i in range(n):
        nG.add_node(i)
    nG.add_edges_from(edgeList)
    return nG


def get_assortativities(step_function, A, m, edge_list, swaps, denominator, G_degree, allow_loops, allow_multi, is_vertex_labeled, total_iterations, last_r):
    t = 0
    r_datapoints = []
    #print("Initially, r = ", nx.degree_pearson_correlation_coefficient(graph_of_graphs.G))
    new_r = last_r[0]
    rejected = np.zeros(1,dtype=np.int64)
    while t < total_iterations: #3*m + 1000:
        rejected[0] = 0        
        step_function(A,edge_list,swaps,rejected,allow_loops,allow_multi)
        delta_r = 0
        new_swap_list = swaps
        if rejected[0] != 1: # A swap was performed
            numerator = G_degree[new_swap_list[0]][1]*G_degree[new_swap_list[2]][1] + G_degree[new_swap_list[1]][1]*G_degree[new_swap_list[3]][1] - G_degree[new_swap_list[0]][1]*G_degree[new_swap_list[1]][1] - G_degree[new_swap_list[2]][1]*G_degree[new_swap_list[3]][1]
            delta_r = 2*numerator*2*m/denominator

        new_r = new_r + delta_r
                
        r_datapoints.append(new_r)
        t+=1
    last_r[0] = new_r   

    #print("Last graph inside function = ", nx.degree_pearson_correlation_coefficient(G2))
    return r_datapoints


def graphs_after_McmcConvergence(step_function, G, A, edge_list, swaps, spacing, allow_loops, allow_multi, is_vertex_labeled, count, has_converged, r_denominator, S2, return_type):
    m = G.number_of_edges()
    n = G.number_of_nodes()      

    if has_converged == False: # Detect convergence if the MCMC is not in its converged region already.
        G_degree = list(nx.degree(G))
        denominator = r_denominator

        S1 = 2*m
        SL = 0
        for e in G.edges():
            SL += 2*G_degree[e[0]][1]*G_degree[e[1]][1]
        numerator = S1*SL - (S2**2)
        r = float(numerator)/denominator # The starting r value is also calculated only once.

        total_iterations = 2*m   
        found = 0
        countchecks = 0
        last_r = [-99]
        last_r[0] = r
        while found == 0:
            test_r = get_assortativities(step_function, A, m, edge_list, swaps, denominator, G_degree, allow_loops, allow_multi, is_vertex_labeled, total_iterations, last_r)
            result = DFGLS(test_r, trend = "c", lags=0)
            pvalue = result.pvalue
            if pvalue < 0.05: # Reject non-stationarity
                found = 1
                break
            del(result)
            
    # At this point, the convergence has been detected.
    graphs_from_configModel = []    
    
    if has_converged == False:
        if return_type == "networkx":
            newgraph = get_networkx_G(n, edge_list, allow_loops, allow_multi)
            graphs_from_configModel.append(newgraph)
        elif return_type == "igraph":
            newgraph = get_igraph_G(n, list(edge_list))
            graphs_from_configModel.append(newgraph)
        else:
            raise ValueError("Incorrect value specified for the argument 'return_type' in the function call.")
        count = count - 1
        has_converged = True
            
    rejected = np.zeros(1,dtype=np.int64)
    for i in range(count):
        for j in range(spacing):
            step_function(A,edge_list,swaps,rejected,allow_loops,allow_multi)
            
        if return_type == "networkx":
            newgraph = get_networkx_G(n, edge_list, allow_loops, allow_multi)
            graphs_from_configModel.append(newgraph)
        elif return_type == "igraph":
            newgraph = get_igraph_G(n, list(edge_list))
            graphs_from_configModel.append(newgraph)
        else:
            raise ValueError("Incorrect value specified for the argument 'return_type' in the function call.")
        
    return graphs_from_configModel, has_converged

def check_density_criterion(G, allow_loops):    
    m = G.number_of_edges()
    n = G.number_of_nodes()
    if allow_loops == True:
        rho = m/(n*n/2)
    else:
        rho = m/((n*(n-1))/2)
    density_factor = 2*rho - rho*rho
    if density_factor <= 0.25:
        return 1
    else:
        return 0
    
def check_maxDegree_criterion(G):
    m = G.number_of_edges()
    degrees = list(nx.degree(G))
    degreeList = []
    for eachtuple in degrees:
        degreeList.append(eachtuple[1])
    sorted_degrees = sorted(degreeList, reverse = True)
    # print(sorted_degrees[0])
    if sorted_degrees[0]*sorted_degrees[0] < (2*m/3): 
        return 1
    else:
        return 0
    
def is_disconnected(deg_seq):
    n = 0
    deg_list = [0]*(max(deg_seq)+1)
    n_unique = 0
    for d in deg_seq:
        if d>0:
            if deg_list[d]==0:
                n_unique+=1 
            n+= 1
            deg_list[d]+=1

    return is_dis_main(deg_list,n_unique,n,deg_seq)
    
def is_dis_main(deg_list,n_unique,n,deg_seq):
    
    deg_list_ori = list(deg_list)
    
    if n==0:
        return False
    
    while deg_list[-1]==0:
        deg_list.pop()


    if n<=2 or len(deg_list)<3:
        return False

    if deg_list[2] == n:
        return True
    
    if len(deg_list)>= n and deg_list[n-1] == n:
        return True

        
    for i in range(0,n+2):
        if deg_list[i]>0:
            min_deg=i
            break

    while n>0:
        
        if n_unique<=2:
            a = min_deg
            b = len(deg_list)-1
            n_a = deg_list[a]
            n_b = deg_list[b]


            if a>=3 and n_a>=3 and a==b-2 and a==n_a+n_b-1 :
                return True

            if a>=3 and n_a>=3 and b-2 == n_a+n_b-1 and a-2==n_b :
                return True
                         
    
        deg_list[min_deg] += -1
        deg_list_ori[min_deg] = 0
        if deg_list[min_deg]==0:
            n_unique += -1
        n += -1
        if min_deg> n:
            return False
    
        s_deg = min_deg 
        prev_s = 0
        i = -1
        while deg_list[i]<= s_deg and s_deg>0:
    
            cur_s = deg_list[i]
            deg_list[i]= prev_s
            if cur_s>0 and prev_s==0:
                n_unique += -1
            if cur_s==0 and prev_s>0:
                n_unique += 1
            s_deg += -cur_s
            prev_s = cur_s
            i += -1
            
         
        if deg_list[i] == 0 and prev_s>0:
            n_unique += 1
            
        deg_list[i] += -s_deg
        deg_list[i] += prev_s
        
        if s_deg>0 and deg_list[i-1]==0:
            n_unique += 1
        deg_list[i-1] += s_deg
     
     
        n += -deg_list[0]
        if deg_list[0]>0:
            n_unique += -1
        deg_list[0] = 0
     
        
        # find new min_deg
        for i in range(0,n+2):              
            if deg_list[i]>0:
                min_deg = i
                break
            
            
        while deg_list[-1]==0:
            deg_list.pop()
            
        if n==0 or len(deg_list)<=3:
            return False

    return False

def autocorrelation_function(series, alpha):
    n = len(series)
    data = np.asarray(series)
    xbar = np.mean(data)
    c0 = np.sum((data - xbar) ** 2)
    
    def standard_autocorrelations(h):
        corr = ((data[: n - h] - xbar) * (data[h:] - xbar)).sum() / c0
        mean = -(n-h)/(n*(n-1))
        var = (n**4 - (h + 3)*n**3 + 3*h*n*n + 2*h*(h+1)*n - 4*h*h)/((n+1)*n*n*(n-1)**2)
        SE = math.sqrt(var)
        standard_corr = (corr - mean)/SE
        return standard_corr

    y = standard_autocorrelations(1) # h = lag = 1
        
    z_critical = stats.norm.ppf(1 - alpha) # One-sided test

    return y, z_critical
    
def get_num_sig_autocorrelations(D, T, r_datapoints, gap, increment, alpha):
    sig = 0
    for i in range(D):
        List_of_r = []
        j = 0
        for k in range(T):
            List_of_r.append(r_datapoints[i][j])
            j += (gap//increment)
        autocorrelation_returned = autocorrelation_function(List_of_r, alpha)
        Rh_value = autocorrelation_returned[0]
        critical_value = autocorrelation_returned[1]
        if Rh_value > critical_value:
            sig += 1
    return sig
    
def progress_D_chains(step_function, T, gap, D, swaps_so_far, last_r, n, m, last_graph, allow_loops, allow_multi, rejected, G_degree, denominator, increment, r_datapoints):
    num_swaps_needed = T*gap - 1
    for i in range(D):
        swaps = swaps_so_far
        counter = 0
        new_r = last_r[i]
        A = np.zeros(shape=(n,n))
        convert_edgelist_to_AdjMatrix(last_graph[i][0], A)              
        while swaps < num_swaps_needed:
            rejected[0] = 0                    
            step_function(A, last_graph[i][0], last_graph[i][1], rejected, allow_loops, allow_multi)
            delta_r = 0
            new_swap_list = last_graph[i][1]
            if rejected[0] != 1: # A swap was performed
                numerator = G_degree[new_swap_list[0]][1]*G_degree[new_swap_list[2]][1] + G_degree[new_swap_list[1]][1]*G_degree[new_swap_list[3]][1] - G_degree[new_swap_list[0]][1]*G_degree[new_swap_list[1]][1] - G_degree[new_swap_list[2]][1]*G_degree[new_swap_list[3]][1]
                delta_r = 2*numerator*2*m/denominator
            new_r = new_r + delta_r

            if counter%increment == 0:
                r_datapoints[i].append(new_r)

            swaps += 1
            counter+= 1
        last_r[i] = new_r    
    
@jit(nopython=True,nogil=True)
def convert_edgelist_to_AdjMatrix(edgeList, A):
    for edge in edgeList:
        A[edge[0], edge[1]] += 1
        A[edge[1], edge[0]] += 1     


@jit(nopython=True,nogil=True)
def check_vertexlabels(nodes):
    for node in nodes:
        if type(node) == "str":
            return 0
        
    return 1

class MCMC:
    '''
    Input
    --------------------
    G (networkx_class) : Graph
    allow_loops (boolean): True if loops are allowed in the graph space. Default is False.
    allow_multi (boolean): True if multiedges are allowed in the graph space. Default is False.
    is_vertex_labeled (boolean): True if choice of graph labelling is vertex-labeled, inconsequential when graph is simple. Default is True.
    verbose (boolean): Use False to silent the warnings in all functions. Default is True. 
                        Warnings of individual functions can be turned on/off by specifying the argument 'verbose' individually in the function parameter.
    
    Returns
    --------------------
    None
    
    Notes
    --------------------
    Raises warnings if properties of G does not match the graph space specified.
    Prints error and exists if G is a weighted or a directed network, or if the graph of graphs in the loopy space is disconnected for the degree sequence of G.
    The default graph space is the vertex-labeled simple graph space.
    '''
    
    def __init__(self, G, allow_loops = False, allow_multi = False, is_vertex_labeled = True, verbose = True):
        self.allow_loops = allow_loops
        self.allow_multi = allow_multi
        self.is_vertex_labeled = is_vertex_labeled
        self.verbose = verbose
        self.has_converged = False
        self.spacing = -1
        self.obtained_spacing = False
        
        if nx.is_weighted(G):
            raise ValueError("Cannot apply double edge swap to weighted networks. Exiting.")
        if G.is_directed():
            raise ValueError("Cannot apply double edge swap to directed networks. Exiting.")
        
        if self.allow_loops == True and self.allow_multi == False: # Loopy graph space
            G_degree = list(nx.degree(G))
            degreeList = []
            for eachpair in G_degree:
                degreeList.append(eachpair[1])
            if is_disconnected(degreeList) == True:
                raise ValueError("The loopy graph space for this degree sequence is disconnected. Please run Nishimura MCMC instead.")
                
        if self.is_vertex_labeled:
            self.step = MCMC_step_vertex
        else:
            self.step = MCMC_step_stub
            
        self.swaps = np.zeros(4,dtype=np.int64)
        
        self.convert_graph(G)

    
    def run_sampling_gap_algorithm(self, T = 500, alpha = 0.04, D = 10, upper_bound = 1, verbose=-999, set_base_1 = -1):   
        '''
        Input
        --------------------
        T (list) : Same size that the sampling gap is calculated from. Default is 500.
        alpha (float) : Significance level for the test of normality of the autocorrelation values. Default is 0.04.
        D (int) : Number of parallel MCMC chains the sampling gap is averaged over. Default is 10.
        upper_bound (int) : Maximum number of MCMC chains that should show significant autocorrelation when the sampling gap is sufficiently large
        verbose (boolean) : Set to True if warning messages are desired particularly for this function. Set to 'False' if warnings are not desired. If not specified, self.verbose is used.
        set_base_1 (int) : Set to 1 when the search for sampling gap starts with a gap of 1, set to 0 otherwise.
        
        Returns
        --------------------
        A sampling gap for the network obtained by running the sampling gap algorithm.
        
        Notes
        --------------------
        If warnings desired, the function prints the value of sig (number of chains currently with significant autocorrelation), upper_bound (target number of chains), and the time elapsed since the start. This is printed every 5 minutes.
        The closer the values of sig and upper_bound, closer is the algorithm to its completion.
        '''
        total_time = 0
        local_verbose = self.verbose
        if type(verbose) == bool:
            local_verbose = verbose

        G_degree = list(nx.degree(self.hashed_G))
        m = self.hashed_G.number_of_edges()
        n = self.hashed_G.number_of_nodes()      

        denominator = self.r_denominator
        
        if self.allow_multi == False:
            base_sampling_gap = int(2*m)
        else:
            if self.is_vertex_labeled == True: 
                base_sampling_gap = int(2.3*m)
            else:
                base_sampling_gap = int(2*m)

        if set_base_1 == 1:
            base_sampling_gap = 1
        increment = int(0.05*m) # The gap is increased by 5% of m with every check.            
        rejected = np.zeros(1,dtype=np.int64)
        if local_verbose==True:
            print("----- Running Burn-in -----\n")
            for j in tqdm(range(1000*m)): # Burn-in period
                self.step(self.A, self.edge_list, self.swaps, rejected, self.allow_loops, self.allow_multi)
            print("----- Burn-in Complete -----")
            print("-- Obtaining Sampling Gap --")
        else:    
            for j in range(1000*m): # Burn-in period
                self.step(self.A, self.edge_list, self.swaps, rejected, self.allow_loops, self.allow_multi)
         
        S1 = 2*m
        S2 = self.S2
        SL = 0
        for e in self.edge_list:
            SL += 2*G_degree[e[0]][1]*G_degree[e[1]][1]
        numerator = S1*SL - (S2**2)
        r_burn_in = float(numerator)/denominator

        eta0_dict = -1
        
        r_datapoints = [[] for i in range(D)]
        for i in range(D):    
            r_datapoints[i].append(r_burn_in)
        last_graph = [] 
        for i in range(D):
            last_graph.append([copy.deepcopy(self.edge_list), copy.deepcopy(self.swaps)])
        last_r = []
        for i in range(D):
            last_r.append(r_burn_in)       
        
        gap = base_sampling_gap

        time1 = time.time()
        swaps_so_far = 0
        while True:
            progress_D_chains(self.step, T, gap, D, swaps_so_far, last_r, n, m, last_graph, self.allow_loops, self.allow_multi, rejected, G_degree, denominator, increment, r_datapoints)
            swaps_so_far = T*gap - 1
                
            sig = get_num_sig_autocorrelations(D, T, r_datapoints, gap, increment, alpha)
            
            if sig <= upper_bound:
                eta0 = gap
                break

            time2 = time.time()
            time_elapsed = (time2 - time1)
            if local_verbose==True and (time_elapsed > 300):
                string = "Significant Autocorrelations = " +str(sig)+", Target = "+str(upper_bound)+", time = "+str(round((total_time+time_elapsed)/3600, 2))+" hour(s).\n"
                print(string)
                total_time = total_time+time_elapsed
                time1 = time.time()
                
            if set_base_1 == 1 and gap == base_sampling_gap:
                gap = increment
            else:
                gap = gap + increment
                
        sampling_gap = eta0
        if local_verbose==True:
            print("- Generating samples from CM -")
        return sampling_gap
   
    def initialise_MCMC_to_G(self): 
        '''
        Input
        --------------------
        None
        
        Returns
        --------------------
        None
        
        Notes
        --------------------
        This function initialises the class variables self.edge_list, self.A and self.swaps w.r.t the original Graph.
        '''
        n = self.hashed_G.number_of_nodes()
        List_edges = []
        for edge in self.hashed_G.edges():                    
            List_edges.append(edge)                    
        self.edge_list = np.array(List_edges)
        
        self.A = np.zeros(shape=(n,n)) # n = number of nodes
        convert_edgelist_to_AdjMatrix(self.edge_list, self.A)
        self.swaps = np.zeros(4,dtype=np.int64)
        
    
    def get_sampling_gap(self, verbose=-999):
        '''
        Input 
        --------------------
        verbose (boolean) : Set to True if warning messages are desired particularly for this function. Set to 'False' if warnings are not desired. If not specified, self.verbose is used.
        
        Returns
        --------------------
        A sampling gap for the network obtained from heuristics when the corresponding constraints are satisfied, or by running the sampling gap algorithm otherwise.
        '''
        
        local_verbose = self.verbose
        if type(verbose) == bool:
            local_verbose = verbose
        
        m = self.hashed_G.number_of_edges()
        n = self.hashed_G.number_of_nodes()
        if self.allow_multi == False:
            density_criterion_satisfied = check_density_criterion(self.hashed_G, self.allow_loops)
            if density_criterion_satisfied == 1:
                return int(2*m)
            else:
                if local_verbose==True:
                    print("The network does not satisfy the density criterion for automatic selection of sampling gap.")
                    print("Running the Sampling Gap Algorithm. This might take a while for large graphs.....")
                sampling_gap = self.run_sampling_gap_algorithm(set_base_1=0, verbose=local_verbose)
                self.initialise_MCMC_to_G()   
                return sampling_gap
        else:
            if self.is_vertex_labeled == False: 
                return int(2*m)
            else:
                maxDegree_criterion_satisfied = check_maxDegree_criterion(self.hashed_G)
                if maxDegree_criterion_satisfied == 1:
                    return int(2.3*m)
                else:
                    if local_verbose==True:
                        print("The network does not satisfy the maximum degree criterion for automatic selection of sampling gap.")
                        print("Running the Sampling Gap Algorithm. This might take a while.....")
                    sampling_gap = self.run_sampling_gap_algorithm(set_base_1=0, verbose=local_verbose)
                    self.initialise_MCMC_to_G()
                    return sampling_gap
    
    def convert_graph(self, G, verbose = -999):
        '''
        Input 
        --------------------
        G (networkx_class) : A networkx graph

        Returns
        --------------------
        None
        
        Notes
        --------------------
        This function initialises several class variables:
            1. original_G (networkx_class): A graph.
            2. hash_map (dictionary): dictionary where key is node label in G and value is the corresponding node number between 0 and n-1.
        reverse_hash_map (dictionary) : A dictionary where key is node number between 0 and n-1 and value is the corresponding node label in G.
            3. reverse_hash_map (dictionary) : A dictionary where key is node number between 0 and n-1 and value is the corresponding node label in G.
            4. hashed_G (networkx_class) : A graph having node labels from 0 to n-1, where n is number of nodes in original_G. It has the same number of nodes and the same edges as present in graph original_G.
            5. edge_list (nx2 numpy array): List of edges in hashed_G. Node labels are integers between 0 to n-1. 
            6. A (nxn numpy array): The adjacency matrix for hashed_G.
            7. r_denominator (float): The denominator of the expression used for calculating the degree assortativity of hashed_G
            8. S2 (float): Sum of square of all degrees in hashed_G
            
        If verbose is True, this function prints warnings if properties of G does not match the graph space specified.
        '''
        local_verbose = self.verbose
        if type(verbose) == bool:
            local_verbose = verbose
        
        self.original_G = G
        
        if self.allow_loops and self.allow_multi:
            self.hashed_G = nx.MultiGraph()
        elif self.allow_loops and not self.allow_multi:
            self.hashed_G = nx.Graph()
        elif not self.allow_multi and not self.allow_loops:
            self.hashed_G = nx.Graph()
        else:
            self.hashed_G = nx.MultiGraph()


        self.hash_map = {} # A dictionary where key = node label in the original network G and value = the corresponding node number between 0 and n-1
        self.reverse_hash_map = {} # A dictionary where key = node number between 0 and n-1 and value = the corresponding node label in the original network G    
        self.metadata_map = defaultdict(dict)
        for node_tuple in G.nodes(data=True):
            for attribute in node_tuple[1]:
                self.metadata_map[attribute][node_tuple[0]] = node_tuple[1][attribute]
        
        
        n = 0
        for node in G.nodes():
            self.hash_map[node] = n
            self.reverse_hash_map[n] = node
            self.hashed_G.add_node(n)
            n+=1

        # Add edges to the new graph
        for eachedge in list(G.edges()):
            self.hashed_G.add_edge(self.hash_map[eachedge[0]], self.hash_map[eachedge[1]])

        list_of_self_loops = list(nx.selfloop_edges(self.hashed_G))
        number_of_loops = len(list_of_self_loops) # Number of self-loops
        if number_of_loops > 0 and self.allow_loops == False:
            if local_verbose == True:
                warnings.warn("Starting graph does not match the specified graph space as it contains self-loops. Removing self-loops from it.")
            for e in list_of_self_loops:
                self.hashed_G.remove_edge(*e)
        if G.is_multigraph() == True and self.allow_multi == False and local_verbose == True:
            warnings.warn("Starting graph G does not match the specified graph space as it contains multi-edges. Converting it to a non-MultiGraph.")

        List_edges = []
        for edge in self.hashed_G.edges():                    
            List_edges.append(edge)                    
        self.edge_list = np.array(List_edges) # Works for both nx.Graph( ) and nx.MultiGraph( ).
        # Note: edge_list = np.array(G.edges()), on the other hand, does not work for nx.MultiGraphs.

        self.A = np.zeros(shape=(n,n)) # n = number of nodes
        convert_edgelist_to_AdjMatrix(self.edge_list, self.A)
        
        G_degree = list(nx.degree(self.hashed_G))
        
        m = self.hashed_G.number_of_edges()

        S1 = 2*m
        S2 = 0
        S3 = 0
        for i in range(n):
            S2 += (G_degree[i][1])**2
            S3 += (G_degree[i][1])**3

        self.r_denominator = S1*S3 - (S2**2)
        self.S2 = S2

    def revert_back(self, G, return_type):
        '''
        Input
        --------------------
        G (networkx_class) : A graph. Node labels of G must be from integer 0 to n-1.
        return_type (str) : Either "networkx" or "igraph" depending on the type of network desired. Using "igraph" is typically much faster than using "networkx".

        Returns
        --------------------
        reverted_G (networkx_class) : A graph of type return_type and with node labels as in the reverse_hash_map.

        '''
        if return_type == "networkx":
            reverted_G = get_empty_nG(self.allow_loops, self.allow_multi)        
            n = G.number_of_nodes()
            for i in range(n):
                reverted_G.add_node(self.reverse_hash_map[i])
            for eachedge in list(G.edges()):
                reverted_G.add_edge(self.reverse_hash_map[eachedge[0]], self.reverse_hash_map[eachedge[1]])
            for attribute in self.metadata_map:
                for node in reverted_G.nodes():
                    reverted_G.nodes[node][attribute] = self.metadata_map[attribute][node]
            
        elif return_type == "igraph":
            reverted_G = ig.Graph(directed=False)
            n = G.vcount()
            reverted_G.add_vertices(n)
            for i in range(len(reverted_G.vs)):
                reverted_G.vs[i]["name"] = self.reverse_hash_map[i]
            for attribute in self.metadata_map:
                for i in range(len(reverted_G.vs)):
                    reverted_G.vs[i][attribute] = self.metadata_map[attribute][self.reverse_hash_map[i]]
            
            rehashed_edgelist = []
            for eachedge in G.get_edgelist():
                rehashed_edgelist.append((self.reverse_hash_map[eachedge[0]], self.reverse_hash_map[eachedge[1]]))
            
            reverted_G.add_edges(rehashed_edgelist)
            
        return reverted_G
    
    def get_graph(self, count=1, verbose = -999, sampling_gap = -999, return_type = "networkx"):
        '''
        Input
        --------------------
        count (int) : Number of graphs to sample from the Configuration model. Default value is 1.
        verbose (boolean) : Set to True if warning messages are desired particularly for functions called by this function. Set to 'False' if warnings are not desired. If not specified, self.warnings is used.
        sampling_gap (int) : Number of double-edge swaps desired (including those rejected) between two MCMC states sampled for the convergence testing.
                             If user provides a sampling_gap, it is used, otherwise the sampling gap is calculated by calling the get_sampling_gap() function
        return_type (str) : Either "networkx" or "igraph" depending on the type of network desired. Using "igraph" is typically much faster than using "networkx".
        
        Returns 
        --------------------
        A graph or a list of graphs with the specified degree sequence, chosen uniformly at random from the desired graph space.
        '''
        if sampling_gap == -999: # if user has not provided a sampling gap
            if self.obtained_spacing == False:
                self.spacing = self.get_sampling_gap(verbose) # obtain sampling gap
                self.obtained_spacing = True
        else:
            self.spacing = sampling_gap # use user-defined sampling gap
                    
        list_of_networks, has_converged = graphs_after_McmcConvergence(self.step, self.hashed_G, self.A, self.edge_list, self.swaps, self.spacing, self.allow_loops, self.allow_multi, self.is_vertex_labeled, count, self.has_converged, self.r_denominator, self.S2, return_type)
        self.has_converged = has_converged         
        
        rehashing_req = 0
        nodes = self.original_G.nodes()
        for node in nodes:
            if type(node) != int: # meaning node label is string
                rehashing_req = 1
                break
                
        if len(self.original_G.nodes(data=True)[0]) != 0: # meaning nodes have attributes
            rehashing_req = 1
                
        if rehashing_req == 0:
            maxNodeLabel = max(nodes)
            if maxNodeLabel != self.original_G.number_of_nodes() - 1:
                rehashing_req = 1
                        
        if rehashing_req == 0:
            if len(list_of_networks) == 1:
                return list_of_networks[0]
            return list_of_networks
        
        else:
            rehashed_networks_from_configModel = []
            for each_net in list_of_networks:
                reverseHashed_net = self.revert_back(each_net, return_type)
                rehashed_networks_from_configModel.append(reverseHashed_net)
    
            if len(rehashed_networks_from_configModel) == 1:
                return rehashed_networks_from_configModel[0]
            return rehashed_networks_from_configModel