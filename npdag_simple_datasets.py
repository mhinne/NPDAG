# sample DAG using Indian Chef's Process

import sys
sys.path.extend(['C:\\Users\\Max\\Dropbox\\Projects\\NPDAG'])
sys.path.extend(['C:\\Users\\u341138\\Dropbox\\Projects\\NPDAG'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['pdf.fonttype'] = 42

import IndianChefProcess
import forwardmodels as fwd
import argparse

import sklearn.datasets as data
import sklearn.linear_model as LogisticRegression

def get_data(dataset, split=0.2):
    
    if dataset == 'breast cancer':
        # load and preprocess data 
        bc_dataset = data.load_breast_cancer()
        X = bc_dataset['data']
        X_bin = 1*(X > np.median(X, axis=0))
        y = bc_dataset['target']
        subset = np.where(y != 2)
        X_bin_subset = X_bin[subset,:][0]
        y_subset = y[subset]
        (n, p) = X_bin_subset.shape        
        
        ixtrain = np.mod(np.arange(0, n),int(1/split))!=int(1/split-1)
        ixtest  = np.invert(ixtrain)
        
        x_train  = X_bin_subset[ixtrain, :]
        x_test   = X_bin_subset[ixtest, :]
        
        y_train  = y_subset[ixtrain]
        y_test   = y_subset[ixtest]
            
        return x_train, x_test, y_train, y_test
    elif dataset == 'iris flowers':
        # load and preprocess data 
        iris_dataset = data.load_iris()
        X = iris_dataset['data']
        X_bin = 1*(X > np.median(X, axis=0))
        y = iris_dataset['target']
        subset = np.where(y != 2) # discard 3rd class
        X_bin_subset = X_bin[subset,:][0]
        y_subset = y[subset]
        (n, p) = X_bin_subset.shape
        
        ixtrain = np.mod(np.arange(0, n),int(1/split))!=int(1/split-1)
        ixtest  = np.invert(ixtrain)
        
        x_train  = X_bin_subset[ixtrain, :]
        x_test   = X_bin_subset[ixtest, :]
        
        y_train  = y_subset[ixtrain]
        y_test   = y_subset[ixtest]
            
        return x_train, x_test, y_train, y_test
    
    elif dataset == 'wine':
        
        wine_dataset = data.load_wine()
        X = wine_dataset['data']
        X_bin = 1*(X > np.median(X, axis=0))
        y = wine_dataset['target']
        subset = np.where(y != 2) # discard 3rd class
        X_bin_subset = X_bin[subset,:][0]
        y_subset = y[subset]
        (n, p) = X_bin_subset.shape
        
        ixtrain = np.mod(np.arange(0, n),int(1/split))!=int(1/split-1)
        ixtest  = np.invert(ixtrain)
        
        x_train  = X_bin_subset[ixtrain, :]
        x_test   = X_bin_subset[ixtest, :]
        
        y_train  = y_subset[ixtrain]
        y_test   = y_subset[ixtest]
                    
        return x_train, x_test, y_train, y_test
    
    elif dataset == 'boolean':
        # This data set we generated ourselves from a predefined DAG
        params = dict()
        params['alpha'] = 1.1 
        params['gamma'] = 1.80
        params['phi']   = 0.1 
        
        # learn some boolean function
        n = 100
        p = 3       
        
        DAG = IndianChefProcess.DAG(params, weights='signed')
        y = DAG.add_node(node=4, observed=True, reputation=0.0, horiz=0.5)
        DAG.add_node(node=5, observed=False, reputation=0.5, horiz=0.35)
        DAG.add_node(node=6, observed=False, reputation=0.5, horiz=0.65)
        DAG.add_edge(5, 4, weight=-1)
        DAG.add_edge(6, 4, weight=1)

        for i in range(p):
            DAG.add_node(node=i+1, observed=True, reputation=1.0, horiz=(i+1)/p-1/(p*2))
        DAG.add_edge(1, 5, weight=1)
        DAG.add_edge(2, 5, weight=-1)
        DAG.add_edge(2, 6, weight=-1)
        DAG.add_edge(3, 6, weight=-1)        
               
        x_train = np.random.choice([0, 1], size=(int(n*(1-split)), p))
        x_test = np.random.choice([0, 1], size=(int(n*split), p))
        
        fwdmodel = fwd.boolean_forward_model(gain=gain)
        y_train = fwdmodel.predict(DAG, x_train)
        y_test = fwdmodel.predict(DAG, x_test)
        
        f = plt.figure()
        ax = f.gca()
        DAG.plot(ax, showlegend=False)
        latex_expression, _, _ = fwdmodel.boolean_expression(DAG)
        ax.set_title(r'Ground truth: ${:s}$'.format(latex_expression), fontdict={'fontsize': 12})
            
        return x_train, x_test, y_train, y_test
    print('Could not load data for {:s}'.format(dataset))
        
    
parser = argparse.ArgumentParser(description='Run ICP sampler for simple data sets.')
parser.add_argument('-niter', dest='niter', nargs='?', type=int, help='Number of MCMC iterations', default=1000)
parser.add_argument('-nchains', dest='nchains', nargs='?', type=int, help='Number of MCMC chains', default=4)
parser.add_argument('-burnin', dest='burnin', nargs='?', type=float, help='Fraction of MCMC iterations to discard', default=0.2)
parser.add_argument('-gain', dest='gain', nargs='?', type=float, help='Logistic gain', default=10)
parser.add_argument('-save', dest='save2file', nargs='?', type=int, help='Save results to disk (1/0)', default=0)

args = parser.parse_args()

gain        = args.gain
nchains     = args.nchains
niter       = args.niter
nburnin     = int(args.burnin*niter)
save2file   = int(args.save2file)

print('Logistic gain in forward model: {:0.2f}'.format(gain))
print('niter: {:d}, nchains: {:d}, nburnin: {:d}'.format(niter, nchains, nburnin))

datasets = ['boolean', 'iris flowers', 'wine', 'breast cancer']


prediction  = lambda DAG, predictors: fwdmodel.predict(DAG, predictors)
accuracy    = lambda prediction, response: np.sum(prediction == response) / len(response)

fwdmodel = fwd.boolean_forward_model(gain=gain)

if save2file: F = open('results_D={:d}_nsamples={:d}.txt'.format(len(datasets), niter),'w')
for dataset in datasets:
    x_train, x_test, y_train, y_test = get_data(dataset)
    _, p = x_train.shape
    print('Dataset \'{:s}\' (p = {:d})'.format(dataset, p))
    
    mcmcresults = IndianChefProcess.sample(fwdmodel,
                                           x_train,
                                           y_train,
                                           niter=niter, 
                                           nchains=nchains,
                                           plotsummary=False, 
                                           verbose=False)
    
    all_scores = list()
    
    for chain in range(nchains):
        scores = [accuracy(prediction(DAG, x_test), y_test) for DAG in mcmcresults[chain]['samples']]
        all_scores.extend(scores[nburnin:])
    
    LR = LogisticRegression.LogisticRegression(solver='lbfgs')
    
    LR_fit = LR.fit(x_train, y_train)
    
    LR_prediction = LR.predict(x_test)
    LR_score = accuracy(LR_prediction, y_test)
    
    if save2file: F.write('Dataset: \'{:s}\' (p = {:d})\n'.format(dataset, p))
    if save2file: F.write('  NPDAG accuracy (BMA): {:0.3f} (SE: {:0.3f})\n'.format(np.mean(all_scores), np.std(all_scores) / np.sqrt(len(all_scores))))
    if save2file: F.write('  Logistic regression accuracy: {:0.3f}\n'.format(LR_score))
    print('  NPDAG accuracy (BMA): {:0.3f} (SE: {:0.3f})'.format(np.mean(all_scores), np.std(all_scores) / np.sqrt(len(all_scores))))
    print('  Logistic regression accuracy: {:0.3f}'.format(LR_score))
if save2file: F.close()

    