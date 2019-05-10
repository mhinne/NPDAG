import numpy as np
import scipy.special as spc
import networkx as nx
import matplotlib.pyplot as plt
import copy

import utility

class DAG():
    
    def __init__(self, params={"alpha": 1, "phi": 1, "gamma": 1}):
        self.nodes              = set()
        self.children           = dict()
        self.parents            = dict()
        self.reputations        = dict()
        self.is_observed        = dict()
        self.xpos               = dict()
        self.values             = dict()
        self.parameters = params
        self.parameters['p'] = self.parameters['phi'] / (self.parameters['alpha'] + self.parameters['phi'])
    
    def copy(self):
        return copy.deepcopy(self)

    def add_node(self, node=None, observations=None, observed=False, reputation=None, rep_lb=None, rep_ub=None, horiz=None):
        pass
        
    # remove node and all references to it
    def remove_node(self, node):
        pass
        
    # add edges - corresponding nodes must already be in DAG    
    def add_edge(self, source, target, weight=None):
        pass
    
    # remove edge and all parent/child references to it    
    def remove_edge(self, source, target):
        pass
        
    def num_active_nodes(self):
        return len(self.get_active_nodes())
    
    def num_edges(self):
        pass
    
    # a node without parents
    def is_orphan(self, node):
        pass
    
    # a node with a single child
    def is_singleton(self, node):
        pass
        
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
        pass
    
    # Find all *latent* nodes with a reputation above target
    def nodes_above(self, target):
        pass
    
    # Get the unique reputation values, for insertion of new nodes
    def get_reputations(self):
        pass
    
    # The log prior probability of the DAG according to the ICP
    # Note that some of the functions in here can induce (slight) numerical instability, even in log-space. The step-wise updates are reliable.
    def log_prior_probability(self):
        pass
    
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
            f = plt.figure()
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
        