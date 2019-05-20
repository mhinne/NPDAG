from functools import reduce
import copy
import random

import chainer
import numpy as np
import scipy.special as spc
import utility
import matplotlib.pyplot as plt

import networkx as nx


class OutputLayer(chainer.Chain):
    def __init__(self, out_size):
        super().__init__()

        with self.init_scope():
            self.linear = chainer.links.Linear(None, out_size)

    def penalization(self):
        return -sum(sum(self.linear.W**2)) if self.linear.W.data is not None else 0

    def __call__(self, x):
        return self.linear(chainer.functions.mean(x, (2, 3)))


class ConvLayer(chainer.Chain):
    def __init__(self, in_size, out_size):
        super().__init__()

        with self.init_scope():
            self.layer = chainer.links.Convolution2D(in_channels=in_size, out_channels=out_size, ksize=3, stride=int(out_size/in_size), pad=1)

    def penalization(self):
        return -sum(sum(sum(sum(self.layer.W**2)))) if self.layer.W.data is not None else 0

    def __call__(self, x):
        return self.layer(x)


class FirstConvLayer(chainer.Chain):
    def __init__(self, in_size, out_size):
        super().__init__()

        with self.init_scope():
            self.layer = chainer.links.Convolution2D(in_channels=in_size, out_channels=out_size, ksize=3, pad=1)

    def penalization(self):
        return -sum(sum(sum(sum(self.layer.W**2)))) if self.layer.W.data is not None else 0

    def __call__(self, x):
        return self.layer(x)


def create_link(parent, child):
    input_size, output_size = get_sizes(child.reputation, parent.reputation)

    if child.reputation == 0:
        #return chainer.links.Linear(None, output_size)
        return OutputLayer(output_size)

    if parent.reputation == 1:
        #return chainer.links.Convolution2D(in_channels=input_size, out_channels=output_size, ksize=3, pad=1)
        return FirstConvLayer(input_size, output_size)
    #return chainer.links.Convolution2D(in_channels=input_size, out_channels=output_size, ksize=3, stride=int(output_size/input_size), pad=1)
    return ConvLayer(input_size, output_size)


def get_sizes(child_reputation, parent_reputation, class_count=10, bin_count=5, offset=1):
    get_size = lambda reputation, k=1: 2 ** (int(bin_count * (1 - reputation)**k) + offset)

    if child_reputation == 0:
        child_size = class_count
    else:
        child_size = get_size(child_reputation)

    if parent_reputation == 1:
        parent_size = None
    else:
        parent_size = get_size(parent_reputation)

    return parent_size, child_size


def ensemble_input(forward_pass, partial_activities, parent_counts, inactive_parents):
    node_activities = {}
    active_nodes = reduce(lambda x,y: x.union(y), map(lambda x: set(x.keys()), forward_pass))
    for node in active_nodes:
        node_partial_activities = [outpt[node]
                                   for outpt in forward_pass
                                   if node in outpt]
        partial_activities[node] += sum(node_partial_activities)
        parent_counts[node] += len(node_partial_activities)
        if parent_counts[node] == node.num_parents - inactive_parents[node]:
            if node.is_target:
                activation = lambda x: x
            else:
                activation = lambda x: chainer.functions.relu(x)
            node_activities[node] = activation(partial_activities.pop(node))
    return node_activities, partial_activities, parent_counts


class Node(object):

    def __init__(self, children, reputation, name, is_observed=False, is_target=False, xpos=None):
        self.children = set(children)
        self.parents = None
        self.reputation = reputation
        self.name = name
        self.is_observed = is_observed
        self.is_target = is_target
        self.data = None
        self.links = {child: create_link(self, child) for child in self.children}
        if xpos is None:
            self.xpos = np.random.uniform(0, 1)

    def __str__(self):
        return self.name

    @property
    def num_children(self):
        return len(self.children)

    @property
    def num_parents(self):
        assert(self.parents is not None)
        return len(self.parents)

    @property
    def is_orphan(self):
        return self.num_parents == 0

    @property
    def is_singleton(self):
        return self.num_children == 1

    @property
    def num_parameters(self):
        return sum([np.prod(link.W.shape) + np.prod(link.b.shape) for link in self.links.values()
                    if link.W.data is not None and link.b.data is not None])

    def observe(self, data, is_target):
        self.is_observed = True
        self.data = data
        if is_target:
            self.is_target = is_target

    @property
    def is_inactive(self):
        active_parents = [p for p in self.parents if not p.is_inactive]
        if active_parents:
            return False
        elif self.is_observed:
            return False
        else:
            return True

    def add_child(self, child):
        self.links.update({child: create_link(self, child)})
        self.children.add(child)

    def remove_child(self, child):
        assert (child in self.children)
        self.links.pop(child)
        self.children.remove(child)

    def __call__(self, inpt):
        return {child: self.links[child](inpt) for child in self.children}


class Network(chainer.ChainList):

    def __init__(self, nodes, prior_parameters={"alpha": 1, "phi": 1, "gamma": 5}):
        names = [node.name for node in nodes]
        assert (sorted(names) == sorted(list(set(names))))
        self.prior_parameters = prior_parameters
        self.prior_parameters['p'] = self.prior_parameters['phi'] / (self.prior_parameters['alpha'] + self.prior_parameters['phi'])
        self.nodes = set(nodes)
        self._update_network()

    @property
    def number_parameters(self):
        return sum([link.W.size + link.b.size for link in self])

    @property
    def parents(self):
        return {node.name: node.parents for node in self.nodes}

    @property
    def children(self):
        return {node.name: node.children for node in self.nodes}

    @property
    def reputations(self):
        return {node.name: node.reputation for node in self.nodes}

    @property
    def is_observed(self):
        return {node.name: node.is_observed for node in self.nodes}

    @property
    def xpos(self):
        return {node.name: node.xpos for node in self.nodes}

    @property
    def num_active_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return sum([len(children) for _, children in self.children.items()])

    @property
    def num_parameters(self):
        return sum([node.num_parameters for node in self.nodes])

    @property
    def num_active_parameters(self):
        return sum([node.num_parameters for node in self.nodes if not node.is_inactive])

    def penalization(self):
        return sum([link.penalization() for link in self.links])

#    def is_orphan(self, node_name):
#        return self.get_node(node_name).is_orphan
#
#    def is_singleton(self, node_name):
#        return self.get_node(node_name).is_singleton

    def copy(self):
        return copy.deepcopy(self)

    def _update_network(self):
        for node in self.nodes:
            node.parents = set([other_node for other_node in self.nodes if node in other_node.children])
        self.links = reduce(lambda u, v: u.union(v), [set(node.links.values()) for node in self.nodes])
        super(Network, self).__init__(*self.links)

    def get_node(self, node_name):
        target_node = [node for node in self.nodes if node.name is node_name][0]
        return target_node

    def add_edge(self, parent_name, child_name):
        # TODO: Assert if connection is there
        parent = self.get_node(parent_name)
        child = self.get_node(child_name)
        parent.add_child(child)
        self._update_network()

    def remove_edge(self, parent_name, child_name):
        # TODO: Assert if connection is not there
        parent = self.get_node(parent_name)
        child = self.get_node(child_name)
        parent.remove_child(child)
        self._update_network()

    def add_node(self, parents_names, children_names, name, reputation=None, rep_lb=0., rep_ub=1., is_observed=False):
        assert ([node.name is not name for node in self.nodes])
        if reputation is None:
            reputation = np.random.uniform(rep_lb, rep_ub)
        parents = set([self.get_node(node_name) for node_name in parents_names])
        children = set([self.get_node(node_name) for node_name in children_names])
        newnode = Node(children=children, reputation=reputation, name=name, is_observed=is_observed)
        self.nodes.add(newnode)
        [self.add_edge(p.name, newnode.name) for p in parents]
        [self.add_edge(newnode.name, c.name) for c in children]
        self._update_network()

    def remove_node(self, node_name):
        node = self.get_node(node_name)
        [self.remove_edge(p.name, node_name) for p in copy.copy(node.parents)]
        [self.remove_edge(node_name, c.name) for c in copy.copy(node.parents)]
        self.nodes.remove(node)
        self._update_network()

    def compute_weight_prior(self, lam=100.):
        return lam*self.penalization()

    def compute_log_lk(self, batch_size):
        dataset_size = [node.data.shape[0] for node in self.nodes if node.is_observed and node.data is not None][0] #TODO: refactor
        indices = random.sample(range(dataset_size), batch_size)
        targets = {node.name: node.data[indices] for node in self.nodes
                   if node.is_target}
        forward_output = self.forward_pass(indices)
        lk_term = -1*dataset_size*sum([chainer.functions.softmax_cross_entropy(x=forward_output[node.name], t=targets[node.name])
                                       for node in self.nodes if node.name in targets])
        prior_term = self.compute_weight_prior()
        #print("Likelihood: {}".format(lk_term.data))
        #print("Prior: {}".format(prior_term.data))
        return lk_term + prior_term

    def compute_model_evidence(self, batch_size):
        dataset_size = [node.data.shape[0] for node in self.nodes if node.is_observed and node.data is not None][
            0]  # TODO: refactor
        log_lk = self.compute_log_lk(batch_size)
        penalization = 0 #self.num_active_parameters #self.num_active_parameters*np.log(dataset_size)/float(2)
        return log_lk - penalization

    def train_parameters(self, num_itr, batch_size, optimizer=chainer.optimizers.Adam(alpha=0.001)):
        optimizer.setup(self)
        loss_list = []
        for itr in range(num_itr):
            self.zerograds()
            loss = -1*self.compute_log_lk(batch_size=batch_size)
            loss.backward()
            optimizer.update()
            loss_list.append(float(loss.data))
        self.loss = loss_list
        #plt.plot(loss_list)
        #plt.savefig("Loss")

    def clear_data(self):
        for node in self.nodes:
            if node.is_observed:
                node.data = None

    def __call__(self, inpt):
        inpt = {self.get_node(name): value for name, value in inpt.items()}
        oupt = inpt
        node_activities = inpt
        parent_counts = {node: 0 for node in self.nodes}
        partial_activities = {node: 0. for node in self.nodes}
        inactive_parents = {node: len([p for p in node.parents if p.is_inactive])
                            for node in self.nodes}
        while set(node_activities.keys()):
            forward_pass = [node(activity)
                            for node, activity in node_activities.items()]
            node_activities, partial_activities, parents_count = ensemble_input(forward_pass,
                                                                                partial_activities,
                                                                                parent_counts,
                                                                                inactive_parents)
            oupt.update(node_activities)
        oupt = {key.name: value for key, value in oupt.items()}
        return oupt

    def forward_pass(self, indices):
        input_dict = {node.name: node.data[indices, :] for node in self.nodes if node.is_observed and not node.is_target}
        return self(input_dict)

    ## Sampler methods ##
        # Find all *latent* nodes with lower reputation -> we force observed nodes at reputation 0 or 1 (input/output)
    def get_nodes_below(self, target_reputation):
        return [node for node in self.nodes if node.reputation < target_reputation]

    def num_nodes_below(self, target_reputation):
        return len(self.get_nodes_below(target_reputation))

    def get_nodes_above(self, target_reputation):
        return [node for node in self.nodes if target_reputation < node.reputation]

    def num_nodes_above(self, target_reputation):
        return len(self.get_nodes_above(target_reputation))

    def get_reputation_intervals(self):
        reputations = sorted(list(set(self.reputations.values()).union({0., 1.})))
        reputation_intervals = list(zip([0.] + reputations, reputations + [1.]))
        return reputation_intervals

    def get_reputation_bounds(self, newparent_reputation):
        reputation_intervals = self.get_reputation_intervals()
        bound = [interval for interval in reputation_intervals
                 if interval[0] <= newparent_reputation < interval[1]]
        return bound[0]

    def log_prior_probability(self):
        alpha = self.prior_parameters['alpha']
        phi = self.prior_parameters['phi']
        gamma = self.prior_parameters['gamma']

        Kplus = self.num_active_nodes
        spc_digamma_alpha = spc.digamma(alpha)

        reputation_intervals = self.get_reputation_intervals()
        reputation_term = -alpha*gamma*sum([(interval[1] - interval[0])*(spc.digamma(alpha + j + 1) - spc_digamma_alpha)
                                            for j, interval in enumerate(reputation_intervals)])

        log_prior = reputation_term - spc.gammaln(Kplus + 1)
        node_terms = []

        for node_k in self.nodes:
            m_k = node_k.num_children
            downstream_k = self.num_nodes_below(node_k.reputation)

            if node_k.is_observed:
                node_terms += [utility.log_pochhammer(phi, m_k) + utility.log_pochhammer(alpha,
                                                                                         downstream_k - m_k) - utility.log_pochhammer(
                    alpha + phi, downstream_k)]
            else:
                assert m_k > 0, "Zero children for latent node {:d}, this should not happen!".format(node_k)
                node_terms += [np.log(alpha) + np.log(gamma) + spc.gammaln(m_k) - utility.log_pochhammer(
                    alpha + downstream_k - m_k, m_k)]

            assert not np.isinf(np.sum(node_terms)), "Infinite log probability - something went wrong!"
        log_prior += np.sum(node_terms)
        return log_prior

    # A random latent node that is open for change.
    def generate_name(self):
        numeric_names = [int(node.name) for node in self.nodes if node.name.isdigit()]
        biggest_name = max(numeric_names) if numeric_names else 0
        return str(biggest_name + 1)

    def get_potential_parents(self, potential_child):
        candidates = self.get_nodes_above(potential_child.reputation)
        potential_parents = [node for node in candidates
                             if not (node.is_singleton and potential_child in node.children)]
        return potential_parents

    # Add or remove an edge
    def resample_graph_connection(self):
        phi = self.prior_parameters['phi']
        alpha = self.prior_parameters['alpha']

        DAG_proposal = self.copy()
        potential_pairs = {node: DAG_proposal.get_potential_parents(potential_child=node) for node in self.nodes
                           if not len(DAG_proposal.get_potential_parents(potential_child=node)) == 0}
        if not potential_pairs:
            return DAG_proposal, -np.inf

        potential_child = random.choice(list(potential_pairs.keys()))
        potential_parent = random.choice(potential_pairs[potential_child])

        m_k = potential_parent.num_children
        m_k_i = m_k if potential_child not in potential_parent.children else m_k - 1
        observation_bonus = phi if potential_parent.is_observed else 0.
        downstream_k = self.num_nodes_below(potential_parent.reputation)
        edge_probability = (m_k_i + observation_bonus)/float(alpha + downstream_k - 1 + observation_bonus)
        edge_sample = np.random.binomial(1, edge_probability)
        if edge_sample == 0 and potential_child in potential_parent.children:
            DAG_proposal.remove_edge(parent_name=potential_parent.name, child_name=potential_child.name)
        elif edge_sample == 1 and potential_child not in potential_parent.children:
            DAG_proposal.add_edge(parent_name=potential_parent.name, child_name=potential_child.name)
        return DAG_proposal, 0.

    # Birth step
    def birth_death_sample(self):
        DAG_proposal = self.copy()
        Kplus = DAG_proposal.num_active_nodes
        candidate_node = random.choice([node for node in DAG_proposal.nodes if not node.reputation == 1])
        Kstar = len([p for p in candidate_node.parents if p.is_singleton and p.is_orphan])
        upstream_i = DAG_proposal.num_nodes_above(candidate_node.reputation)
        sad_parents = [p for p in candidate_node.parents
                       if p.is_singleton and p.is_orphan and not p.is_observed]

        if not sad_parents or np.random.binomial(1, 0.5) == 1:
            newparent_reputation = np.random.uniform(low=candidate_node.reputation, high=1.)
            reputation_bounds = self.get_reputation_bounds(newparent_reputation)
            newparent_name = self.generate_name()
            DAG_proposal.add_node(parents_names={}, children_names={candidate_node.name},
                                  name=newparent_name, reputation=newparent_reputation)
            asymmetry_correction = np.log((reputation_bounds[1] - reputation_bounds[0])*(upstream_i + 1)*Kplus) -np.log(Kstar + 1)
        else:
            sacrificial_parent = np.random.choice(sad_parents)
            reputation_bounds = self.get_reputation_bounds(sacrificial_parent.reputation)
            DAG_proposal.remove_node(node_name=sacrificial_parent.name)
            asymmetry_correction = np.log(Kplus) - np.log((reputation_bounds[1] - reputation_bounds[0])*(Kplus-1)*upstream_i)

        log_prior_ratio = DAG_proposal.log_prior_probability() - self.log_prior_probability()
        return DAG_proposal, log_prior_ratio + asymmetry_correction

    def resample_reputation(self):
        DAG_proposal = self.copy()
        possible_nodes = [node for node in DAG_proposal.nodes if not node.is_observed]
        if not possible_nodes:
            return DAG_proposal, -np.inf
        candidate_node = random.choice(possible_nodes)
        candidate_name = candidate_node.name
        candidate_parents = candidate_node.parents
        candidate_children = candidate_node.children
        upper_reputation_bound = 1. if not candidate_node.parents else np.min([node.reputation for node in candidate_node.parents])
        lower_reputation_bound = 0. if not candidate_node.children else np.max([node.reputation for node in candidate_node.children])
        sampled_reputation = np.random.uniform(lower_reputation_bound, upper_reputation_bound)
        DAG_proposal.remove_node(node_name=candidate_name)
        DAG_proposal.add_node(parents_names=[p.name for p in candidate_parents],
                              children_names=[c.name for c in candidate_children],
                              name=candidate_name, reputation=sampled_reputation)
        log_prior_ratio = DAG_proposal.log_prior_probability() - self.log_prior_probability()
        return DAG_proposal, log_prior_ratio


    # Plot methods
    def plot(self, ax=None, showlegend=True, save_name=None, show=False):
        nxG = nx.DiGraph()
        for node in self.nodes:
            nxG.add_node(node.name)

        edgelabels = dict()
        for fromnode in self.nodes:
            if fromnode.name in self.children:
                for tonode in self.children[fromnode.name]:
                    nxG.add_edge(fromnode.name, tonode.name)
                    #edgekey = '{:d}_{:d}'.format(fromnode.name, tonode.name)

        nodelabels = dict()
        positions = dict()
        for node in self.nodes:
            positions[node.name] = [node.xpos, node.reputation]
            nodelabels[node.name] = node.name
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
            ax.plot([0], [0], color=colormap[label], label=label)

        nx.draw_networkx(nxG, ax=ax, pos=positions, arrows=True, nodelist=nodelist,
                         node_color=colors, labels=nodelabels, width=2, arrowsize=25, arrowstyle='->')
        nx.draw_networkx_edge_labels(nxG, ax=ax, edge_labels=edgelabels, pos=positions, font_size=20)

        ax.set_ylim(-0.2, 1.2)
        ax.set_ylabel('Reputation')
        ax.set_xlabel('Arbitrary horizontal coordinate')
        ax.set_xlim(0., 1.)

        if showlegend: ax.legend(['Observed (fixed)', 'Observed', 'Latent'])

        if save_name:
            plt.savefig(save_name)

        if show:
            plt.show()

class NetworkSampler(object):

    def __init__(self, initial_network, num_itr, batch_size, evidence_batch_size):
        initial_network.train_parameters(num_itr=num_itr, batch_size=batch_size) #TODO: work in progress
        self.network_samples = [copy.deepcopy(initial_network)]
        self.batch_size = batch_size
        self.num_itr = num_itr
        self.evidence_batch_size = evidence_batch_size

    # Network training
    def update_network(self, num_samples):
        current_sample = self.network_samples[-1]
        sample_index = 0
        while sample_index < num_samples:
            for sampler in [current_sample.resample_graph_connection, current_sample.birth_death_sample, current_sample.resample_reputation]:

                try:
                    DAG_proposal, log_acceptance_ratio = sampler()

                    DAG_proposal.train_parameters(self.num_itr, self.batch_size)
                    proposed_log_model_evidence = float(DAG_proposal.compute_model_evidence(self.evidence_batch_size).data)
                    current_log_model_evidence = float(current_sample.compute_model_evidence(self.evidence_batch_size).data)
                    corrected_log_acceptance_ratio = log_acceptance_ratio + proposed_log_model_evidence - current_log_model_evidence
                    acceptance_ratio = np.exp(corrected_log_acceptance_ratio)
                    #print(acceptance_ratio) # TODO: Debug
                    if acceptance_ratio >= 1 or np.random.binomial(1, acceptance_ratio) == 1:
                        #sample = copy.deepcopy(DAG_proposal)
                        #sample.clear_data()
                        #self.network_samples.append(sample)
                        current_sample = DAG_proposal
                        #DAG_proposal.plot(save_name='figures/Sample {}.png'.format(sample_index), show=False)
                        sample_index += 1
                        #print(sample_index) #DEBUG
                except Exception as error:
                    print(error.message)
        return current_sample




## Dataset ##
number_pixels = 28*28
number_output_classes = 10
train, test = chainer.datasets.get_mnist()
train_in, train_target = zip(*train)
train_in = chainer.Variable(np.stack([np.reshape(image, newshape=(1, 28, 28))
                                      for image in train_in]).astype("float32"))
train_target = chainer.Variable(np.stack([tg.astype("int32")
                                          for tg in train_target]))


## Initial network ##
a = Node(children=set(), reputation=0., name="a")
l = Node(children={a}, reputation=0.2, name="l")
e = Node(children={l}, reputation=1., name="e")
network = Network({a, e, l})
a.observe(data=train_target, is_target=True)
e.observe(data=train_in, is_target=False)

## Test model ##

## Training ##
#network.train_parameters(num_itr=500, batch_size=50)

sampler = NetworkSampler(initial_network=network, num_itr=400, batch_size=100, evidence_batch_size=10000)
sampler.update_network(num_samples=500)

network.plot()
plt.show()

print("Reputation intervals: " + str(list(network.get_reputation_intervals())))
print("Num edges: " + str(network.num_edges))
print("Log prior: " + str(network.log_prior_probability()))

batch_size = 10
im_size = 28
num_channels = 1
x = {"c": chainer.Variable(np.random.normal(0, 1, (batch_size, num_channels, im_size, im_size)).astype("float32")),
     "d": chainer.Variable(np.random.normal(0, 1, (batch_size, num_channels, im_size, im_size)).astype("float32"))}

outpt = network(x)
print([n + ": " + str(val.shape) for n, val in outpt.items()])

