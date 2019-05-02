import numpy as np
import scipy.special as spc
import networkx as nx
import matplotlib.pyplot as plt
import copy

import utility

def sample(fwdmodel, predictors, response, DAG_init=None, niter=100, nchains=1, verbose=False, plotsummary=False):
    # Sample from the Indian Chef's Process using Metropolis MCMC.     
    predictors = utility.array2dict(predictors)
    
    p = len(predictors)
    if DAG_init is None:
        # Initialize with DAG in which all predictors connect to all responses, weights are uniform random
        params = dict()
        params['alpha'] = 1.1 # alpha > 1, see eq (13); large values makes DAGs very crowded
        params['gamma'] = 1.80
        params['phi']   = 0.1
        
        DAG_init = list()
        for chain in range(nchains):
            D = DAG(params, weights='signed')
            X = []
            for i in range(p):
                X += [D.add_node(node=i+1, observed=True, reputation=1.0,horiz=(i+1)/p-1/(p*2))]
            y = D.add_node(node=p+1, observed=True, reputation=0.0, horiz=0.5)
            for x in X:
                D.add_edge(x, y)
            DAG_init += [D]
    
    
    if nchains > 1:
        assert isinstance(DAG_init, list) and len(DAG_init) == nchains, 'Each chain must have its own initialization!'
    if nchains == 1:
        if not isinstance(DAG_init, list):
            DAG_init = [DAG_init]
        
    results = dict()
    for chain in range(nchains):
        results[chain] = dict()
        
        samples = [DAG_init[chain]]         # running sample
        iter_samples = [DAG_init[chain]]    # final sample per iteration
        
        log_prior           = [DAG_init[chain].log_prior_probability()]
        log_likelihood      = [fwdmodel.log_likelihood(DAG_init[chain], predictors, response)]
        iter_log_prior      = [log_prior[-1]]
        iter_log_likelihood = [log_likelihood[-1]]
        
        n_accepts = 0
        n_rejects = 0
        
        n_jumps = 0
        
        # Start sampling
        for t in range(1, niter):
            if verbose: print("Iteration {:d}".format(t))
            DAG_current = samples[-1]
                            
            # 1: resample graph connections
            node_i = DAG_current.get_potential_child()   
            if verbose: print('Phase 1')
            potential_parents_i = DAG_current.get_potential_parents(node_i)
            
            for node_k in potential_parents_i:
                DAG_current = samples[-1]
                DAG_proposal, delta_P, rstype = DAG_current.resample_graph_connection(node_i, node_k) 
               
                L_proposal = fwdmodel.log_likelihood(DAG_proposal, predictors, response)
                L_current = log_likelihood[-1]
                delta_L = L_proposal - L_current
                log_alpha = delta_P + delta_L  
                           
                if np.random.rand() < np.min([np.exp(log_alpha), 1.0]):            
                    if rstype is 'removal':
                        if verbose: print('Removed edge ({:d}, {:d})'.format(node_k, node_i))    
                    if rstype is 'addition':
                        if verbose: print('Added edge ({:d}, {:d})'.format(node_k, node_i))    
                    DAG_current = DAG_proposal
                    log_prior += [log_prior[-1] + delta_P]
                    log_likelihood += [L_proposal]
                    n_accepts += 1
                else:
                    n_rejects += 1
                    log_prior += [log_prior[-1]]
                    log_likelihood += [L_current]
                
                samples += [DAG_current]
                
                
            # 2: birth/death nodes
            if verbose: print('Phase 2')
            # Note that delta_P contains a reversible-jump Jacobian term
            DAG_proposal, delta_P, bdtype, node = DAG_current.birth_death_moves()  
            # In some cases nothing can be added, i.e. when a death would isolate another node; often happens in initial samples
            if node is not None:            
                L_proposal = fwdmodel.log_likelihood(DAG_proposal, predictors, response)
                L_current = log_likelihood[-1]
                delta_L = L_proposal - L_current
                log_alpha = delta_P + delta_L             
                if np.random.rand() < np.min([np.exp(log_alpha), 1.0]):
                    if node is None:
                        print('Accepted none move')
                    else:
                        if verbose: print('Accepted {:s} of node {:d}'.format(bdtype, node))
                    DAG_current = DAG_proposal
                    log_prior += [DAG_proposal.log_prior_probability()]
                    log_likelihood += [L_proposal]
                    n_accepts += 1
                    n_jumps += 1
                else:
                    n_rejects += 1
                    log_prior += [DAG_current.log_prior_probability()]
                    log_likelihood += [L_current]
                
                samples += [DAG_current]
    
            
            # 3: reorder reputation; typically likelihood does not depend on this step
            if verbose: print('Phase 3')
            DAG_proposal, delta_P, rotype, node = DAG_current.reorder_reputations()
            L_proposal = fwdmodel.log_likelihood(DAG_proposal, predictors, response)
            L_current = log_likelihood[-1]
            delta_L = L_proposal - L_current
            log_alpha = delta_P + delta_L 
            if np.random.rand() < np.min([np.exp(log_alpha), 1.0]):              
                if verbose: print('Accepted {:s} of node {:d}'.format(rotype, node))
                DAG_current = DAG_proposal
                log_prior += [log_prior[-1] + delta_P]
                log_likelihood += [L_proposal]
                n_accepts += 1
            else:
                n_rejects += 1
                log_prior += [log_prior[-1]]
                log_likelihood += [L_current]
            
            samples += [DAG_current]            
            
            iter_samples += [DAG_current]
            iter_log_prior += [log_prior[-1]]
            iter_log_likelihood += [log_likelihood[-1]]
           
        if plotsummary:
            
            edgecount   = lambda DAG: np.sum([len(DAG.children[node]) for node in DAG.nodes if node in DAG.children])
            nodecount   = lambda DAG: len(DAG.nodes)
            num_edges   = [edgecount(sample) for sample in iter_samples]
            num_nodes   = [nodecount(sample) for sample in iter_samples]
            f, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
            axes[0,0].plot(iter_log_prior)
            axes[0,0].set_title('Log prior')
            axes[0,0].set_xlabel('Iteration')
            axes[0,1].plot(iter_log_likelihood)
            axes[0,1].set_title('Log likelihood')
            axes[0,1].set_xlabel('Iteration')
            axes[1,0].plot(num_nodes)
            axes[1,0].set_title('Node count')
            axes[1,0].set_xlabel('Iteration')    
            axes[1,1].plot(num_edges)
            axes[1,1].set_title('Edge count')
            axes[1,1].set_xlabel('Iteration')   
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.suptitle('MCMC summary', fontsize=16)
            
        if verbose: print('Number of reversible jump steps accepted: {:d}'.format(n_jumps))    
        arate = n_accepts / (n_accepts+n_rejects)
        if verbose: print('Acceptance rate: {:0.3f}'.format(arate))
        results[chain]['samples']           = iter_samples
        results[chain]['log_prior']         = iter_log_prior
        results[chain]['log_likelihood']    = iter_log_likelihood
        results[chain]['acceptance_rate']   = arate
        results[chain]['initialization']    = DAG_init[chain]
    if nchains == 1:
        results = results[0]
    return results


class DAG():
    
    def __init__(self, params, weights=None):
        # set of integers
        self.nodes              = set()
        self.parameters         = dict()
        self.pos                = dict()
        self.n_obs              = None
        self.children           = dict()
        self.parents            = dict()
        self.reputations        = dict()
        self.is_observed        = dict()
        self.xpos               = dict()
        self.values             = dict() # can be observed or derived
        if weights is not None:
            self.weighttype     = weights
            self.edgeweights    = dict() 
        else:
            self.weighttype     = None
        
        if params is not None:
            self.parameters = params
        else:        
            self.parameters['alpha']    = 1 / np.random.gamma(0.5, 0.5)
            self.parameters['phi']      = np.random.gamma(0.5, 0.5)            
            self.parameters['gamma']    = np.random.gamma(0.5, 0.5)        
        self.parameters['p'] = self.parameters['phi'] / (self.parameters['alpha'] + self.parameters['phi'])
    
    def copy(self):
        return copy.deepcopy(self)
    ### Graph operations
    
    # initialize new node
    def add_node(self, node=None, observations=None, observed=False, reputation=None, rep_lb=None, rep_ub=None, horiz=None):
        if node is None:
            if len(self.nodes) is 0:
                node = 1
            else:
                node = np.max([node for node in self.nodes])+1
        assert node not in self.nodes, "This node already exists!"
        self.nodes.add(node)
        if reputation is not None:
            self.reputations[node] = reputation
        elif rep_lb is not None and rep_ub is not None:
            self.reputations[node] = np.random.uniform(low=rep_lb, high=rep_ub)
        else:
            self.reputations[node] = np.random.uniform(low=0.0, high=1.0)
        if observations is not None:
            self.values[node] = observations
            self.is_observed[node] = True
        else:
            self.is_observed[node] = observed
        if horiz is None:
            self.xpos[node] = np.random.uniform(0.1, 0.9)
        else:
            self.xpos[node] = horiz        
        return node
        
    # remove node and all references to it
    def remove_node(self, node):
        assert node in self.nodes, "This node is not in the DAG"        
        if node in self.children:
            for child in self.children[node]:
                self.parents[child].remove(node)
        if node in self.parents:
            for parent in self.parents[node]:
                self.children[parent].remove(node)
        self.nodes.remove(node)
        self.reputations.pop(node)        
        self.xpos.pop(node)
        if self.is_observed[node]:            
            self.values.pop(node)
        self.is_observed.pop(node)
        self.parents.pop(node, None)
        self.children.pop(node, None)
        
    # add edges - corresponding nodes must already be in DAG    
    def add_edge(self, source, target, weight=None):
        assert source in self.nodes, "Source node unknown!"
        assert target in self.nodes, "Target node unknown!"
        assert self.reputations[source] > self.reputations[target], "Edges must follow reputation ordering"
        if source not in self.children:
            self.children[source] = set()
        if target not in self.parents:
            self.parents[target] = set()
        self.children[source].add(target)
        self.parents[target].add(source)
        
        if self.weighttype is not None:
            edgekey = '{:d}_{:d}'.format(source, target)
            edgekey = '{:d}_{:d}'.format(source, target)
            if self.weighttype is 'signed':
                if weight is None:
                    self.edgeweights[edgekey] = np.random.choice([-1, 1], size=1)[0]
                else:
                    self.edgeweights[edgekey] = weight
            if self.weighttype is 'boolean':
                if weight is None:
                    self.edgeweights[edgekey] = np.random.choice(['not', ''], size=1)[0]
                else:
                    self.edgeweights[edgekey] = weight
    
    # remove edge and all parent/child references to it    
    def remove_edge(self, source, target):
        assert source in self.nodes, "Source node does not exist in DAG"
        assert target in self.nodes, "Target node does not exist in DAG"
        self.children[source].remove(target)        
        if len(self.children[source]) is 0:
            self.children.pop(source)
            
        self.parents[target].remove(source)
        if len(self.parents[target]) is 0:
            self.parents.pop(target)
        if self.weighttype is not None:
            edgekey = '{:d}_{:d}'.format(source, target)
            self.edgeweights.pop(edgekey)
    
    # weight of an edge - edge must exist in DAG    
    def get_weight(self, source, target):
        if self.weighttype is None:
            return target in self.children[source]
        edgekey = '{:d}_{:d}'.format(source, target)
        assert edgekey in self.edgeweights, "This edge does not exist in DAG"
        return self.edgeweights[edgekey]
    
    # all nodes in the current DAG
    def get_active_nodes(self):
        return self.nodes
        
    def num_active_nodes(self):
        return len(self.get_active_nodes())
    
    def num_edges(self):
        M = 0
        for child_set in self.children.values():
            M += len(child_set)
        return M
    
    # a node without parents
    def is_orphan(self, node):
        return node not in self.parents
    
    # a node with a single child
    def is_singleton(self, node):
        return len(self.children[node]) == 1
        
    ### Development utility
    # Some helper functions for implementations    
    def get_input_nodes(self):
        return {node for node in self.nodes if self.reputations[node] == 1.0}
    
    def get_response_nodes(self):
        return {node for node in self.nodes if self.reputations[node] == 0.0}
    
    def get_latent_nodes(self):
        return {node for node in self.nodes if not self.reputations[node] in(0.0, 1.0)}
    
    def get_observations(self, node):
        return self.values[node]
    
    def get_reputation(self, node):
        return self.reputations[node]
    

    ### ICP sampler functions

    # Find all *latent* nodes with lower reputation -> we force observed nodes at reputation 0 or 1 (input/output)
    def nodes_below(self, target):
        out_nodes = []        
        for node in self.nodes:
            if self.reputations[node] < self.reputations[target] and self.reputations[node] > 0:
                out_nodes += [node]        
        return out_nodes
    
    # Find all *latent* nodes with a reputation lower than target reputation
    def nodes_below_rep(self, target_rep):
        out_nodes = []
        for node in self.nodes:
            if self.reputations[node] < target_rep and not self.reputations[node] > 0:
                out_nodes += [node]
        return out_nodes
    
    # Find all *latent* nodes with a reputation above target
    def nodes_above(self, target):
        out_nodes = []        
        for node in self.nodes:
            if self.reputations[node] > self.reputations[target] and self.reputations[node] < 1.0:
                out_nodes += [node]        
        return out_nodes
    
    # Get the unique reputation values, for insertion of new nodes
    def get_reputations(self):
        reputations = [self.reputations[node] for node in self.nodes if not self.reputations[node] in (0.0, 1.0)]
        reputations += [0.]
        reputations += [1.]
        reputations = np.unique(reputations)
        reputations = np.sort(reputations)
        return reputations
    
    # The log prior probability of the DAG according to the ICP
    # Note that some of the functions in here can induce (slight) numerical instability, even in log-space. The step-wise updates are reliable.
    def log_prior_probability(self):
        alpha   = self.parameters['alpha']
        phi     = self.parameters['phi']
        gamma   = self.parameters['gamma']
        
        Kplus   = self.num_active_nodes() 
        
        spc_digamma_alpha = spc.digamma(alpha)
        
        reputations = self.get_reputations()
        reputation_term = 0
        for j in range(len(reputations)-1):
            reputation_term += (reputations[j+1] - reputations[j]) * (spc.digamma(alpha + j) - spc_digamma_alpha)
        reputation_term *= -alpha*gamma
                
        log_prior = reputation_term - spc.gammaln(Kplus+1)
        node_terms = []
        
        for node_k in self.nodes:
            m_k = len(self.children[node_k]) if node_k in self.children else 0
            downstream_k = len([node for node in self.nodes if self.reputations[node] < self.reputations[node_k]])
           
            if self.is_observed[node_k]:
                node_terms += [utility.log_pochhammer(phi, m_k) + utility.log_pochhammer(alpha, downstream_k - m_k) - utility.log_pochhammer(alpha+phi, downstream_k)]
            else:
                assert m_k > 0, "Zero children for latent node {:d}, this should not happen!".format(node_k)
                node_terms += [np.log(alpha) + np.log(gamma) + spc.gammaln(m_k) - utility.log_pochhammer(alpha + downstream_k - m_k, m_k)]
                
            assert not np.isinf(np.sum(node_terms)), "Infinite log probability - something went wrong!"
        log_prior += np.sum(node_terms)
        return log_prior
    
    # A random latent node that is open for change.
    def get_random_nonstatic_node(self):        
        candidate_nodes = [node for node in self.nodes if not self.reputations[node] in (0.0, 1.0)] 
        if len(candidate_nodes) > 0:
            return np.random.choice(candidate_nodes)
        else:
            return None
        
    # the response node(s) can be selected!    
    def get_potential_child(self):
        candidate_nodes = [node for node in self.nodes if self.reputations[node] < 1.0]
        if len(candidate_nodes) > 0:
            return np.random.choice(candidate_nodes)
        else:
            return None
        
    def get_potential_parents(self, node_i):
        candidates = []        
        for node in self.nodes:
            if self.reputations[node] > self.reputations[node_i]:
                candidates += [node]                  
        potential_parents = []
        for node_k in candidates:
            if node_k in self.children:
                nb_k = self.children[node_k]
                if len(nb_k) > 1:
                    potential_parents += [node_k]
                elif len(nb_k) is 1: # in this case we need to check whether removal of connection removes node k from DAG
                    if not node_i in nb_k: # node i is not a child of k
                        potential_parents += [node_k]
        return potential_parents    
    
    # Add or remove an edge
    def resample_graph_connection(self, node_i, node_k):
        phi             = self.parameters['phi']
        alpha           = self.parameters['alpha']
    
        DAG_proposal    = copy.deepcopy(self)
        
        m_k = len(self.children[node_k]) if node_k in self.children else 0
        downstream_k = len([node for node in self.nodes if self.reputations[node] < self.reputations[node_k]])  
                
        if node_k in self.children and not node_i in self.children[node_k]:
            # edge addition; (k,i) \not \in E
            DAG_proposal.add_edge(node_k, node_i)
            p_accept = np.log((phi*self.is_observed[node_k] + m_k) / (alpha + downstream_k - m_k - 1))
            return DAG_proposal, p_accept, 'addition'
        else:
            # edge removal; (k,i) \in E
            p_accept = np.log((alpha + downstream_k - m_k) / (phi*self.is_observed[node_k] + m_k - 1))            
            DAG_proposal.remove_edge(node_k, node_i)
            return DAG_proposal, p_accept, 'removal'        
    
    # Add or remove a node
    def birth_death_moves(self):
        node_i = self.get_potential_child()  
        
        # those nodes that connect to i only, and have no parents themselves
        singleton_orphan_parents_i = []
        
        if node_i in self.parents:
            parents_of_i = self.parents[node_i]
        else:
            parents_of_i = []
        for parent in parents_of_i:
            if self.is_orphan(parent) and self.is_singleton(parent):
                singleton_orphan_parents_i += [parent]
        n_singleton_orphan_parents_i = len(singleton_orphan_parents_i)
        
        log_prior_DAG_current = self.log_prior_probability()
        nodes_above_i = self.nodes_above(node_i)
        
        Kplus = self.num_active_nodes()
               
        DAG_proposal = copy.deepcopy(self)
        # uniformly choose between birth and death of nodes
        move = 'birth' if np.random.rand() < 0.5 else 'death'
        if move is 'birth':  
            # birth of a new node           
            reputations = [self.reputations[node_i]]
            for node_j in nodes_above_i:                    
                reputations += [self.reputations[node_j]]
            reputations += [1.0]
            reputations = np.sort(reputations) 
            
            rep_interval = np.random.randint(len(reputations)-1)
            rep_lb = reputations[rep_interval]
            rep_ub = reputations[rep_interval+1]
            
            node_k = DAG_proposal.add_node(rep_lb=rep_lb, rep_ub=rep_ub, observed=False) 
            
            DAG_proposal.add_edge(node_k, node_i)  
            
            log_prior_DAG_proposal = DAG_proposal.log_prior_probability()

            hastings_factor = np.log(rep_ub - rep_lb) \
                        + np.log(len(nodes_above_i)+1) + np.log(Kplus) - np.log(n_singleton_orphan_parents_i + 1)

            log_p_accept_birth = log_prior_DAG_proposal - log_prior_DAG_current \
                                    + hastings_factor                                        
                                                    
            # Valar dohaeris
            return DAG_proposal, log_p_accept_birth, 'birth', node_k 
        else:
            # death of a dynamic singleton-orphan parent            
            dynamic_singleton_orphan_parents_i = [node for node in singleton_orphan_parents_i if self.reputations[node] < 1.0]
            n_dynamic_singleton_orphan_parents_i = len(dynamic_singleton_orphan_parents_i)
            
            
            if n_dynamic_singleton_orphan_parents_i > 0:
                node_k = np.random.choice(dynamic_singleton_orphan_parents_i)
                reps = list(self.get_reputations())
                ix = reps.index(self.reputations[node_k])
                rep_below = reps[ix-1]
                rep_above = reps[ix+1]    
                
                DAG_proposal.remove_node(node_k)
                                
                log_prior_DAG_proposal = DAG_proposal.log_prior_probability()
                
                hastings_factor = np.log(n_dynamic_singleton_orphan_parents_i) - np.log(rep_above - rep_below) - np.log(Kplus - 1) - np.log(len(nodes_above_i))
                
                log_p_accept_death = log_prior_DAG_proposal - log_prior_DAG_current \
                                    + hastings_factor
                
                # Valar morghulis                    
                return DAG_proposal, log_p_accept_death, 'death', node_k
            else:
                return self, 0.0, 'none', None # always accept this

    # Sample new reputation values within the interval dictated by parents and children    
    def reorder_reputations(self):        
        DAG_proposal = copy.deepcopy(self)
        node_i = DAG_proposal.get_random_nonstatic_node()
        
        if node_i is not None:    
            parents_i = self.parents[node_i] if node_i in self.parents else []
            parents_reps = [self.reputations[node] for node in parents_i]
            parents_reps += [1.0]
            children_i = self.children[node_i] if node_i in self.children else []
            children_reps = [self.reputations[node] for node in children_i]
            children_reps += [0.0]
            
            rep_lb = np.max(children_reps)
            rep_ub = np.min(parents_reps)
            
            rep_i_new = np.random.uniform(low=rep_lb, high=rep_ub)
            DAG_proposal.reputations[node_i] = rep_i_new
            
            log_p_accept_reorder = DAG_proposal.log_prior_probability() - self.log_prior_probability() 
            return DAG_proposal, log_p_accept_reorder, 'reordering', node_i
        else:
            return self, 0.0, 'nochange', None
        
    ### visualization & output    
    def print_node(self, node):
        print("Node {:d}".format(node))
        print("  Rep.: {:0.3f}, Vis.: {:s}".format(self.reputations[node], 'O' if self.is_observed[node] else 'H'))
        if node in self.parents:
            parent_string = ''
            for parent in self.parents[node]:
                parent_string += '{:d} '.format(parent)
            print('  Parents: ( {:s})'.format(parent_string))
        if node in self.children:
            child_string = ''
            for child in self.children[node]:
                child_string += '{:d} '.format(child)
            print('  Children: ( {:s})'.format(child_string))
        
    def plot(self, ax=None, showlegend=True):
        nxG = nx.DiGraph()
        for node in self.nodes:
            nxG.add_node(node)
        
        edgelabels = dict()
        for fromnode in self.nodes:
            if fromnode in self.children:
                for tonode in self.children[fromnode]:
                    nxG.add_edge(fromnode, tonode)
                    edgekey = '{:d}_{:d}'.format(fromnode, tonode)
                    w = self.edgeweights[edgekey]
                    if self.weighttype is 'signed':
                        if w > 0:
                            label = ''
                        else:
                            label = r'$\neg$'    
                        edgelabels[(fromnode, tonode)] = label
                    else:
                        edgelabels[(fromnode, tonode)] = w
        
        nodelabels = dict()
        positions = dict()
        for node in self.nodes:
            positions[node] = [self.xpos[node], self.reputations[node]]
            nodelabels[node] = node
        nodelist = nxG.nodes()
        
        colormap = {'fixed': 'y', 'observed': 'r', 'latent': 'g'}
        colors = []
        for node in nodelist:
            if self.reputations[node] in (0.0, 1.0):
                colors += [colormap['fixed']]
            else:
                if self.is_observed[node]:
                    colors += [colormap['observed']]
                else:
                    colors += [colormap['latent']]
        if ax is None:
            f = plt.figure();
            ax = f.gca()            
            
        for label in colormap.keys():
            ax.plot([0],[0],color=colormap[label],label=label)
        
        nx.draw_networkx(nxG, ax=ax, pos=positions, arrows=True, nodelist=nodelist, 
                         node_color=colors, labels=nodelabels, width=2, arrowsize=25, arrowstyle='->')        
        nx.draw_networkx_edge_labels(nxG, ax=ax, edge_labels=edgelabels, pos=positions, font_size=20)
        
        ax.set_ylim(-0.2, 1.2)
        ax.set_ylabel('Reputation')
        ax.set_xlabel('Arbitrary horizontal coordinate')
        ax.set_xlim(0., 1.)
        
        if showlegend: ax.legend(['Observed (fixed)', 'Observed', 'Latent'])
        