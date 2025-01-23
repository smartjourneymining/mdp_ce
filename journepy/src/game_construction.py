import networkx as nx
import json 
import numpy as np 

def ms(trace):
    """Concats the trace to a multiset-history

    Args:
        trace (List[Dict]): Trace to convert to ms

    Returns:
        Str: json encoding of sorted multiset
    """
    multiset = {}
    for pos in trace:
        if pos['concept:name'] not in multiset:
            multiset[pos['concept:name']] = 1
        else:
            multiset[pos['concept:name']] += 1
    return json.dumps(multiset, sort_keys=True).encode().decode("utf-8") # use json encodings for multisets

def sequence(trace):
    """Computes the sequence-history of the given trace

    Args:
        trace (List[Dict]): Trace to convert to sequence

    Returns:
        Str: " - " concatenated string of sequence
    """
    hist = str(trace[0]['concept:name'])
    for pos in trace[1:]:
        hist += " - " + str(pos['concept:name']) # construct history
    return hist

def transition_system(log, history, refinement):
    """Function to compute a transition system, given a pre-processed log

    Args:
        log (List[List[Dict]]): _description_
        history (int): History to consider in each state
        refinement (function): Refinement function

    Returns:
        networkx.DiGraph, Dict[edges -> traces]: Transition system encoded as directed graph, dict mapping edges to their tranversing traces
    """
    edges = []
    edge_counter = {}
    controll = {}
    action = {}
    edge_mapping = {}
    for trace_index in range(len(log)):
        trace = log[trace_index]
        s = "start"
        assert(trace[0]['concept:name']=="start")
        for pos_index in range(1,len(trace)):
            pos = trace[pos_index]
            activity = pos['concept:name']
            #t = ms(trace[max(0,pos_index-history+1):pos_index+1])
            t = refinement(trace[max(0,pos_index-history+1):pos_index+1])
            e = (s,t)
            action[e] = activity
            if e not in edges:
                edges.append(e)
                edge_counter[e] = 1
                edge_mapping[e] = [trace_index]
            else:
                edge_counter[e] = edge_counter[e]+1
                edge_mapping[e].append(trace_index)
            s = t
    g = nx.DiGraph()
    for e in edges:
        g.add_edge(e[0], e[1])
    to_remove = [] # to remove selve-loops
    for e in g.edges:
        if e[0] == e[1]:
            to_remove.append(e)
        # set properties
        g[e[0]][e[1]]['action'] = action[e]

    for e in to_remove:
        if e in g.edges():
            g.remove_edge(e[0],e[1])
    
    return g, edge_mapping

def isInTrace(s,t, trace):
    """Checks if transition s to to t occurs in trace

    Args:
        s (Str): Start node
        t (Str): Target node
        trace (List[Dict]): Considered trace 

    Returns:
        Bool: True if transition is found, else false
    """
    for i in range(len(trace)-1):
        if trace[i]['concept:name'] == s and trace[i+1]['concept:name'] == t:
            return True
    return False

def weight(trace):
    """_COmputes weight of trace, 1 if pos, -1 if neg
    """
    return 1 if any("positive" in pos['concept:name'] for pos in trace) else -1

def entropy(p1, p2):
    """Entropy over prop. distributions p1 an p2.

    Args:
        p1 (Float):
        p2 (Float):

    Returns:
        Float: Entropy
    """
    if p1 == 0 or p2 == 0:
        return 0
    return - p1*np.log2(p1) - p2* np.log2(p2)

def distribution(s,t,log, edge_mapping):
    """Computes distribution over positive and negative edges  by using constructed edge mapping from `transition_system`

    Args:
        s (Str): Start state
        t (Str): Target state
        log (List[List[Dict]]): Input log
        edge_mapping (): 2nd result from transition_system

    Returns:
        Dict[]: Mapping 1 and -1 to the number of their occurences
    """
    distr = {1.0: 0 , -1.0 : 0}
    assert((s,t) in edge_mapping)
    for trace_index in edge_mapping[(s,t)]:
        w = weight(log[trace_index])
        distr[w] += 1 #
    return distr[1], distr[-1]

def compute_edge_cost(g, log, edge_mapping, bias = -0.21):
    """Edge cost per transition

    Args:
        g (networkx.DiGraph): Input graph
        log (List[List[Dict]]): Input log in xes - list format
        edge_mapping (): 2nd result from transition_system

    Returns:
        _type_: _description_
    """
    edge_cost = {}
    counter = 1
    for s in g.nodes:
        counter +=1
        for t in g[s]:

            
            p1, p2 = distribution(s,t,log, edge_mapping)
            w = 1 if p1 >= p2 else -1

            wp1 = p1/(p1+p2)
            wp2 = p2/(p1+p2)

            scaling = 20
            entro = entropy(wp1, wp2)

            edge_cost[(s,t)] = (((1-entro) * w) + bias )*scaling

    return edge_cost

def annotate_graph(g, edge_cost):
    for e in edge_cost:
        g[e[0]][e[1]]['cost'] = round(edge_cost[e],2)
    return g

def add_traversal_information(g, edge_mapping):
    for e in g.edges:
        g.edges[e]['edge_traversal'] = len(edge_mapping[e])

    for s in g:
        if s not in ["start"] and "pos" not in s and "neg" not in s:
            assert(sum( [g.edges[e]['edge_traversal'] for e in g.in_edges(s)] ) == sum( [g.edges[e]['edge_traversal'] for e in g.out_edges(s)] ))
        outgoing_sum = sum( [g.edges[e]['edge_traversal'] for e in g.out_edges(s)] )
        if "pos" in s or "neg" in s:
            outgoing_sum = sum( [g.edges[e]['edge_traversal'] for e in g.in_edges(s)] ) # change to ingoing sum
        g.nodes[s]['node_traversal'] = outgoing_sum

    return g

def model(log, history, refinement):
    """Computes transition system based on log, given history and refinement.

    Args:
        log (List[List[Dict]]): Input log
        history (int): State history
        refinement (function): Refinement function pointer

    Returns:
        networkx.DiGraph: Annotated transition system(cost, node_traversal)
    """
    system, edge_mapping = transition_system(log, history, refinement)
    edge_cost = compute_edge_cost(system, log, edge_mapping)
    g = annotate_graph(system, edge_cost)
    g = add_traversal_information(g, edge_mapping)
    return g


def load_actors(path):
    """Reads and returns json-dict actor dict

    Args:
        path (Str): Path to actor file

    Returns:
        Dict: Mapping activities to actors
    """
    with open(path) as f:
        data = f.read()
    actors = json.loads(data)
    return actors

def annotate_actors(g, actors):
    """Annotates actors to transition system

    Args:
        g (networkx.DiGraph): Input graph
        actors (Dict): Mapping activities to ['company', 'customer']
    """
    for e in g.edges:
        controllable_set = False
        for key in actors:
            if key in g.edges[e]['action']:
                controllable_set = True
                g.edges[e]['controllable'] = actors[key] == 'company'
        if not controllable_set:
            g.edges[e]['controllable'] = True

def game(log, history, refinement, actors={}, actor_path=""):
    """ Main function to build game. Constructs game by building transition system first and then extending with actors.

    Args:
        log (List[List[Dict]]): Input log
        history (int): History to be considered in each state.
        refinement (function): Function pointer for refinement definition
        actors (dict, optional): Actor mapping. Defaults to {}.
        actor_path (str, optional): Actor path. Defaults to "".

    Returns:
        _type_: _description_
    """
    g = model(log, history, refinement)
    if actor_path == "":
        annotate_actors(g, actors)
    else:
        annotate_actors(g, load_actors(actor_path))
    return g