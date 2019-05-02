import utility as util
import numpy as np

class boolean_forward_model:
    # This semi-boolean model combines parents' input via potential negations (for negative edge weights) and taking the median value of the input.
    # The forward model is a sigmoid, which approaches a deterministic model for large gain.
    
    def __init__(self, gain=1):
        self.gain = gain

    def boolean_NOT(self, x, w):
        if w < 0:
            return 1-x
        else:
            return x
        
    # Special AND gate
    def boolean_MEDIAN(self, x):
        if x.ndim == 1:
            return x
        else:
            return 1*(np.sum(x-0.5, axis=0) > 0)
    
    def boolean_AND(self, x):
        if x.ndim == 1:
            return x
        else:
            return np.prod(x, axis=0)
    
    def boolean_OR(self, x):
        if x.ndim == 1:
            return x
        else:
            n, = x.shape
            return (np.sum(x, axis=0)>-n)*2-1

    def log_likelihood(self, DAG, predictors, response, epsilon=1e-3):        
        if not isinstance(predictors, dict):
            predictors = util.array2dict(predictors)
            
        X = DAG.get_input_nodes()
        Z = DAG.get_latent_nodes()
        Y = DAG.get_response_nodes()
        
        zvalues = dict()
        # We assume only 1 response node
        y = next(iter(Y))
        
        n = len(response)
        
        # highest reputation first
        Z_by_rep = sorted(Z, key=DAG.get_reputation, reverse=True)    
        
        for z in Z_by_rep:            
            if not z in DAG.parents:
                zvalues[z] = np.zeros(shape=(n,))
            else:
                parents = DAG.parents[z]
                parent_values = np.zeros((len(parents), n))
                for ix, i in enumerate(parents):
                    w = DAG.get_weight(i, z)
                    if i in X:
                        parent_values[ix, :] = self.boolean_NOT(predictors[i], w)
                    else:
                        parent_values[ix, :] = self.boolean_NOT(zvalues[i], w)
                
                parent_values = np.array(parent_values)                
                zvalue = self.boolean_MEDIAN(parent_values)
                zvalues[z] = zvalue
        
        parents_y = DAG.parents[y]
        
        parent_y_values = np.zeros((len(parents_y), n))
        for ix, i in enumerate(parents_y):
            w = DAG.get_weight(i, y)
            if i in X:
                parent_y_values[ix, :] = self.boolean_NOT(predictors[i], w)
            elif i in Z:
                parent_y_values[ix, :] = self.boolean_NOT(zvalues[i], w)
        
        if parents_y == 1:
            theta = util.sigmoid(2*parent_y_values-1, gain=self.gain)
        else:
            theta = util.sigmoid(2*np.mean(parent_y_values, axis=0)-1, gain=self.gain)
        L = np.sum(response*np.log(theta) + (1 - response)*np.log(1 - theta))
        return L
        
    
    def predict(self, DAG, predictors):        
        if not isinstance(predictors, dict):
            predictors = util.array2dict(predictors)
        
        Z = DAG.get_latent_nodes()
        Y = DAG.get_response_nodes()
        X = DAG.get_input_nodes()
        
        non_input_nodes = Z.union(Y)
        non_input_nodes_by_rep = sorted(non_input_nodes, key=DAG.get_reputation, reverse=True)
        
        x = next(iter(X))
        n = len(predictors[x])
        
        node_values = dict()
        
        for node in non_input_nodes_by_rep:
            if node in DAG.parents:
                parents = DAG.parents[node]
                weighted_parent_values = np.zeros((len(parents), n)) 
                for ix, parent in enumerate(parents):
                    if parent in X:
                        parvals = predictors[parent]
                    else:
                        parvals = node_values[parent]
                    w = DAG.get_weight(parent, node)
                    weighted_parent_values[ix, :] += self.boolean_NOT(parvals, w)
                node_values[node] = self.boolean_MEDIAN(weighted_parent_values)
            else: 
                node_values[node] = np.zeros((n))
                      
        y = next(iter(Y))
        
        parents_y = DAG.parents[y]
        
        parent_y_values = np.zeros((len(parents_y), n))
        for ix, i in enumerate(parents_y):
            w = DAG.get_weight(i, y)
            if i in X:
                parent_y_values[ix, :] = self.boolean_NOT(predictors[i], w)
            elif i in Z:
                parent_y_values[ix, :] = self.boolean_NOT(node_values[i], w)
        

        if parents_y == 1:
            theta = util.sigmoid(2*parent_y_values-1, gain=self.gain)
        else:            
            theta = util.sigmoid(2*np.mean(parent_y_values, axis=0)-1, gain=self.gain)
        
        response = 1*(np.random.rand(n) < theta) 
        return response
        
    
    def boolean_expression(self, DAG):
        # return expression in conjunctive normal form        
        def flatten(test_list):
            if isinstance(test_list, list):
                if len(test_list) == 0:
                    return []
                first, rest = test_list[0], test_list[1:]
                return flatten(first) + flatten(rest)
            else:
                return [test_list]
        
        def get_parent_expression(node):
            if node not in DAG.parents:
                return str(node)
            else:
                sub_expr = '('
                for i, parent in enumerate(sorted(DAG.parents[node])):
                    if i > 0:
                        sub_expr += ' AND'
                    w = DAG.get_weight(parent, node)
                    if w == -1:
                        sub_expr += ' NOT ' + get_parent_expression(parent)
                    else:
                        sub_expr += ' ' + get_parent_expression(parent)
                sub_expr += ' )'
                return sub_expr
            
        def get_parent_list(node, w):
            if node not in DAG.parents:
                return [w*node]
            else:
                sub_list = []
                for parent in sorted(DAG.parents[node]):
                    wpar = w*DAG.get_weight(parent, node)
                    if w == -1:
                        sub_list += [get_parent_list(parent, wpar)]
                    else:
                        sub_list += [get_parent_list(parent, wpar)]
                return sub_list    
                    
        Y = DAG.get_response_nodes()
        X = DAG.get_input_nodes()
        
        y = next(iter(Y))
        expression = 'y := ' + get_parent_expression(y) 
        expr_list = get_parent_list(y, 1)
        
        expr_list = flatten(expr_list)
        expr_set = set(expr_list)
        
        for m in X:
            if m in expr_set and -m in expr_set:
                expr_set.remove(m)
                expr_set.remove(-m)
        
        simplified_expr = ''
        for m in X:
            if m in expr_set:
                if len(simplified_expr) > 0:
                    simplified_expr += ' AND ' + str(m)
                else:
                    simplified_expr += str(m)
            elif -m in expr_set:
                if len(simplified_expr) > 0:
                    simplified_expr += ' AND NOT ' + str(m)
                else:
                    simplified_expr += 'NOT ' + str(m)

        latex_expression = simplified_expr.replace('AND', r'\wedge')
        latex_expression = latex_expression.replace('NOT', r'\neg')
        if len(latex_expression) == 0:
            latex_expression = r'\bot'
        
        return latex_expression, simplified_expr, expression
        
        