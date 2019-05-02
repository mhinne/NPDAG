import numpy as np
import time

def log_pochhammer(x, n):
    res = 0
    for i in range(n):
        res += np.log(x + i)
    return res

def print_node_list(nodes):
    for node in nodes:
        print(node)
		
def sigmoid(x, L=1, gain=1, x0=0):
	return L / (1+np.exp(-gain*(x - x0)))

# Courtesy of https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python: 
def tic():
    #Homemade version of matlab tic and toc functions    
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(verbose = False):
    if 'startTime_for_tictoc' in globals():
        elapsed = round(time.time() - startTime_for_tictoc,2)
        if verbose:
            print("Elapsed time is " + str(elapsed) + " seconds.")
        return elapsed
    else:
        print("Toc: start time not set")
        
def array2dict(array):
    (n, p) = array.shape    
    array_dict = dict()    
    for i in range(p):
        array_dict[i+1] = array[:, i]    
    return array_dict