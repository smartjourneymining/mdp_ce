import networkx as nx
import copy
from .mc_utils.uppaal_utils import * 
import subprocess

def shifted_lists(l):
    """Computes all possible shift of the given list.

    Args:
        l (List): Input list of cycle elements

    Returns:
        List[List]: List of lists enumerating all list shifts
    """
    shifted_lists = []
    for j in range(len(l)):
        list_constructed = copy.deepcopy(l[j:])
        list_constructed.extend(l[:j])
        list_constructed.append(list_constructed[0])
        shifted_lists.append(list_constructed)
    return shifted_lists

def contains(hist, c):
    """Checks if history hist contains circle c

    Args:
        hist (List[Str]): History is list of strings (names of states visited so far)
        c (List[Str]): Cycle as list

    Returns:
        int: count of how often cycles c was completed in any way in history.
    """
    n = len(c)+1
    max_count = 0
    lists = shifted_lists(c)
    for helper_list in lists:
        count = 0
        for i in range(len(hist)-(n-1)):
            if hist[i:i+n] == helper_list:
                count += 1
        max_count = max(max_count, count)
    return max_count

def is_on(e,v,c):
    """Returns true if edge (e,v) is on c

    Args:
        e (Str): start
        v (Str): target
        c (List[Str]): Cycle

    Returns:
        bool: Returns True if edge (e,v) lies on cycle c
    """
    for i in range(len(c)-1):
        if c[i] == e and c[i+1] == v:
            return True
    if c[-1] == e and c[0] == v:
        return True
    
def unroll(G, start, target, k, debug = False):
    """Presented Unrolling algorithm, Algorithm 1 with online reducing

    Args:
        G (netwrokx.DiGraph): Input transition system as directed graph
        start (Str): Start state
        target (Str): Target state
        k (int): max number of cycle iterations
        debug (bool, optional): If set to True, additional information is printed. Defaults to False.

    Returns:
        networkx.DiGraph: Unrolled transition system
    """
    G_gen = nx.DiGraph()
    G_gen.add_node(start, hist = [str(start)])
    if 'controllable' in G.nodes[start]:
        G_gen.nodes[start]["controllable"] = G.nodes[start]["controllable"]

    cycles = list(nx.simple_cycles(G))

    queue = [start]
    # start bf-search
    while(queue):
        if debug:
            print(len(G_gen.nodes), len(queue))
        s = queue[0]
        queue.pop(0)
        s_original = str(s).split("vv")[0]
        neighbours = list(G[s_original])
        for t in neighbours:
            t_original = t
            local_hist = copy.deepcopy(G_gen.nodes[s]["hist"])
            local_hist.append(str(t_original))
            is_on_cycle = False
            can_traverse = False
            path = []
            circle = []
            relevant_cycle = []
            for c in cycles:
                if is_on(s_original,t_original,c):
                    relevant_cycle.append(c)
                    
            all_smaller = True
            for c in relevant_cycle:
                if contains(local_hist,c) >= k:
                    all_smaller = False
            
            if not all_smaller:
                paths = list(nx.all_simple_paths(G, source=t, target=target))
                for p in paths:
                    merged_hist = copy.deepcopy(local_hist)
                    merged_hist.extend(p[1:]) # 1.st element already added
                    can_not_traverse = False
                    
                    #test if no loop larger than k with path
                    for c_loop in relevant_cycle:
                        if contains(merged_hist,c_loop) > k : # check that there is path without completing additional cycle
                            can_not_traverse = True
                    can_traverse = not can_not_traverse
            if all_smaller or can_traverse:               
                #every node not on cycle can be unqiue ("merge point" within unrolled graph)
                if relevant_cycle:
                    while t in G_gen.nodes:
                        if "vv" not in t:
                            t += "vv1"
                        else:
                            t = t.split("vv")[0]+"vv"+str(int(t.split("vv")[-1])+1)
                # add node t only to graph if not already treated

                if t not in queue:
                    queue.append(t)
                    G_gen.add_node(t, hist = local_hist)
                assert(s in G_gen and t in G_gen)
                G_gen.add_edge(s,t)
                if('cost' in G[s_original][t_original]):
                    G_gen[s][t]['cost'] = G[s_original][t_original]['cost']
                if('controllable' in G[s_original][t_original]):
                    G_gen[s][t]['controllable'] = G[s_original][t_original]['controllable']

    return G_gen

def query(g, query_path, uppaal_stratego, output, unrolling_factor = 0, debug = False):
    """ Computes mapping R from alg. 1
        Return results too for easier handling

    Args:
        g (_type_): _description_
        query_path (_type_): _description_
        uppaal_stratego (_type_): _description_
        output (_type_): _description_
        unrolling_factor (int, optional): _description_. Defaults to 0.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    """"""
    # partial graph implications, per activity
    results = {}
    
    assert('start' in g)
    dfs_numbers = {}

    for n,i in zip(nx.dfs_postorder_nodes(g, 'start'), range(len(g.nodes)-1)):
        dfs_numbers[i] = n
    dfs_numbers[len(g.nodes)-1] = 'start'

    for i in range(len(g.nodes)):
    #for a in g.nodes:
        assert(i in dfs_numbers)
        current_state = dfs_numbers[i]

        sub_nodes = set()
        sub_nodes.update(set(list(nx.descendants(g, current_state))))
        sub_nodes.add(current_state) # cant assume that all sub_nodes are contained due to loops

        all_neighbours_contained = True
        neighbour_results = []
        for n in g[current_state]:
            if n not in results:
                all_neighbours_contained = False
                break
            neighbour_results.append(results[n])

        all_leaves_computed = True
        leave_results = []
        for s in sub_nodes:
            if g.out_degree(s) == 0 and s != current_state:
                if s not in results:
                    all_leaves_computed = False
                    break
                leave_results.append(results[s])
        
        # can set results if all neighbours are contained
        if all_neighbours_contained and len(set(neighbour_results))==1:
            results[current_state] =  neighbour_results[0]# write to assign result if determined
            g.nodes[current_state]["positive_guarantee"] = neighbour_results[0]
            continue
        
        # can set result by reachable leaves
        if all_leaves_computed and len(set(leave_results)) == 1:
            results[current_state] = leave_results[0]
            g.nodes[current_state]["positive_guarantee"] = leave_results[0]
            continue

        """
        states = [a]

        sub_nodes = set()
        for s in states:
            sub_nodes.update(set(list(nx.descendants(g, s))))
            sub_nodes.add(s)
        if len(sub_nodes) > args.threshold_reachable_nodes: # execute only on nodes with less than threshold descendants for better performance
            continue

        """
        subgraph = nx.subgraph(g, sub_nodes)
        subgraph = nx.DiGraph(subgraph)

        # add start node to subgraph
        start_nodes = []
        for n in subgraph.nodes:
            if subgraph.in_degree(n) == 0:
                start_nodes.append(n)
        for n in start_nodes:
            subgraph.add_edge("start", n)
            subgraph["start"][n]["controllable"] = True
            subgraph["start"][n]["cost"] = 0
        # if initial node lies on cycle, per default set as start node
        if "start" not in subgraph.nodes:
            subgraph.add_edge("start", current_state)
            subgraph["start"][current_state]["controllable"] = True
            subgraph["start"][current_state]["cost"] = 0

        if debug:
            nx.write_gexf(subgraph, output+"test.gexf")
            to_uppaal(subgraph, output+'bpi2017subgraph.xml')
        target = [s for s in subgraph.nodes if "positive" in s or "negative" in s]
        subgraph_unrolled = unroll(subgraph, "start", target, unrolling_factor)
        positives = []
        for s in subgraph_unrolled.nodes:
            if "positive" in s:
                positives.append(s)
        #assert(len(positives) <= 1)
        to_uppaal(subgraph_unrolled, output+'bpi2017subgraph.xml')
        out = subprocess.Popen([uppaal_stratego, output+'bpi2017subgraph.xml', query_path], stdout=subprocess.PIPE)
        out.wait()
        result = "is satisfied" in str(out.communicate()[0])
        results[current_state] = result

        g.nodes[current_state]["positive_guarantee"] = result
    
    return g, results

# Function to compute clusters for decision boundary
def reachable_cluster(g, results):
    pos_cluster = []
    neg_cluster = []
    g_copy = copy.deepcopy(g)
    for s in g:
        subgraph = nx.subgraph(g, set(list(nx.descendants(g, s))))
        subgraph = nx.DiGraph(subgraph)
        nodes = [s for s in subgraph]
        sub_results = [results[n] for n in results if n in nodes]
        g.nodes[s]['reducible'] = False
        if len(set(sub_results))<2:
            g.nodes[s]['reducible'] = True
            # sub_results has size 0 or 1: 0 if end node, 1 else
            if results[s]:
                pos_cluster.append(s)
            else:
                neg_cluster.append(s)

    for s in pos_cluster:
        g_copy = nx.contracted_nodes(g_copy, "pos", s)
    for s in neg_cluster:
        g_copy = nx.contracted_nodes(g_copy, "neg", s)
   
    g_copy.remove_edges_from(nx.selfloop_edges(g_copy))

    db = []
    for s in g_copy.nodes:
        pos = False
        neg = False
        if s not in g:
            # after contraction
            continue
        for n in g_copy[s]:
            if "pos" in n:
                pos = True
            if "neg" in n:
                neg = True
        
        if pos and neg and len(g_copy[s]) == 2:
            assert(s in g.nodes)
            g.nodes[s]['decision_boundary'] = True
            db.append(s)
            #g.nodes[s]['viz'] = {'color': {'r': 0, 'g': 0, 'b':255, 'a': 0}}
            g.nodes[s]['color'] = "blue"
            g.nodes[s]['shape'] = "box"
            g.nodes[s]['viz'] = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 0}}
        else:
            assert(s in g.nodes)
            g.nodes[s]['decision_boundary'] = False
    
    return g, db

def game_db(g, results):
    g_copy = copy.deepcopy(g)
    positive_cluster = []
    for s in results:
        if results[s]:
            g_copy.nodes[s]['color'] = "green"
            positive_cluster.append(s)

    # attempted merge:
    negative_cluster = []
    for s in g_copy.nodes:
        reachable = set(list(nx.descendants(g_copy, s)))
        reaches_pos = False
        for neighbour in reachable:
            if "pos" in neighbour:
                reaches_pos = True
        if "pos" in s:
            reaches_pos = True
        if not reaches_pos:
            g_copy.nodes[s]['color'] = "red"
            negative_cluster.append(s)
    for s in negative_cluster:
        g_copy = nx.contracted_nodes(g_copy, "neg", s)
    for s in positive_cluster:
        g_copy = nx.contracted_nodes(g_copy, "pos", s)

    db = []
    for s in g:
        pos = False
        neg = False
        g.nodes[s]['decision_boundary'] = False
        g.nodes[s]['positive_guarantee'] = results[s]
        if s in g_copy:
            for n in g_copy[s]:
                if "pos" in n:
                    pos = True
                if "neg" in n:
                    neg = True
            if pos and neg and len(g_copy[s]) == 2:
                db.append(s)
                g.nodes[s]['decision_boundary'] = True
                g.nodes[s]['color'] = "blue"
                g.nodes[s]['shape'] = "box"
                g.nodes[s]['viz'] = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 0}}

    return g, db

# Uses the decision boundary computation as model reduction
def db_reduction(g, static = False):
    pos_cluster = []
    neg_cluster = []
    if not static:
        for s in g.nodes:
            if g.nodes[s]['positive_guarantee']:
                g.nodes[s]['color'] = "green"
                pos_cluster.append(s)

        for s in g.nodes:
            reachable = set(list(nx.descendants(g, s)))
            reaches_pos = False
            for neighbour in reachable:
                if "pos" in neighbour:
                    reaches_pos = True
            if "pos" in s:
                reaches_pos = True
            if not reaches_pos:
                g.nodes[s]['color'] = "red"
                neg_cluster.append(s)
    else:
        for s in g:
            subgraph = nx.subgraph(g, set(list(nx.descendants(g, s))))
            subgraph = nx.DiGraph(subgraph)
            nodes = [s for s in subgraph]
            sub_results = [g.nodes[n]["positive_guarantee"] for n in nodes if 'positive_guarantee' in g.nodes[n]]
            if len(set(sub_results))<2:
                # sub_results has size 0 or 1: 0 if end node, 1 else
                if g.nodes[s]["positive_guarantee"]:
                    pos_cluster.append(s)
                else:
                    neg_cluster.append(s)

    for s in pos_cluster:
        g = nx.contracted_nodes(g, "pos", s)
    for s in neg_cluster:
        g = nx.contracted_nodes(g, "neg", s)
    g.nodes['pos']['decision_boundary'] = False
    g.nodes['neg']['decision_boundary'] = False
    g.nodes['pos']['positive_guarantee'] = True
    g.nodes['neg']['positive_guarantee'] = False

    g.nodes["pos"]['viz'] = {'color': {'r': 0, 'g': 255, 'b': 0, 'a': 0}, 'position' : {'x' : 0.0, 'y': 0.0, 'z': 0.0}}
    g.nodes["neg"]['viz'] = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 0}, 'position' : {'x' : 0.0, 'y': 0.0, 'z': 0.0}}

    g.nodes["neg"]["final"] = True
    g.nodes["pos"]["final"] = True

    for s in g:
        if "final" not in g.nodes[s]:
             g.nodes[s]["final"] = False

    g.remove_edges_from(nx.selfloop_edges(g))
    return g

def compute_db(g, query_path, uppaal_stratego, output, static = False, unrolling_factor = 0, debug = False):
    g, results = query(g, query_path, uppaal_stratego, output, unrolling_factor = unrolling_factor, debug = debug)
    if not static:
        g, db = game_db(g, results)
    else:
        g, db = reachable_cluster(g, results)
    return g, db