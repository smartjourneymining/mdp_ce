import argparse
from src.preprocessing import bpic17, preprocessing_utils
from src import game_construction, game_operations, game_operations
import bpi_utils
from src.mc_utils import uppaal_utils
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np
import copy

SEQUENCE_HISTORY = 3

# distribuion over positive and negative traces in log
def compare_outcomes(log):
    outcome = ["positive" if "positive" in [i['concept:name'] for i in trace] else "negative" for trace in log]
    print(Counter(outcome))

def investigate_log(log):
    counts_positive = {"W_Call incomplete" : 0, "W_Call after":0}
    pos_count = 0
    for trace in log:
        if preprocessing_utils.contains(trace, "positive") :
            pos_count += 1
            update = 0
            for e in trace:
                if "W_Call incomplete" in e['concept:name']:
                    counts_positive["W_Call incomplete"] += 1
                    update += 1
                if "W_Call after" in e['concept:name']:
                    counts_positive['W_Call after'] += 1

    counts_negative = {"W_Call incomplete" : 0, "W_Call after":0}
    neg_count = 0
    for trace in log:
        if preprocessing_utils.contains(trace, "negative") :
            neg_count += 1
            for e in trace:
                if "W_Call incomplete" in e['concept:name']:
                    counts_negative["W_Call incomplete"] += 1
                if "W_Call after" in e['concept:name']:
                    counts_negative['W_Call after'] += 1

    both = {"W_Call incomplete" : 0, "W_Call after":0}
    for k in counts_positive:
        both[k] = (counts_negative[k] + counts_positive[k])/(neg_count+pos_count)

    for k in counts_positive:
        counts_positive[k] /= pos_count
    for k in counts_negative:
        counts_negative[k] /= neg_count

    print("Average number of calls in successful journeys", counts_positive)
    print("Average number of calls in unsuccessful journeys", counts_negative)
    print("Average number of calls in all journeys", both)

    counts_pos = []
    counts_neg = []
    for trace in log:
        count = 0
        if preprocessing_utils.contains(trace, "positive") :
            for e in trace:
                if "O_Create Offer" in e['concept:name']:
                    count += 1
            counts_pos.append(count)
        if preprocessing_utils.contains(trace, "negative"):
            for e in trace:
                if "O_Create Offer" in e['concept:name']:
                    count += 1
            counts_neg.append(count)
    print("Average number of created offers in successful journeys", sum(counts_pos)/len(counts_pos))
    print("Average number of created offers in unsuccessful journeys", sum(counts_neg)/len(counts_neg))

    counts = []
    for trace in log:
        count = 0
        for e in trace:
            if "O_Create Offer" in e['concept:name']:
                count += 1
        counts.append(count)
    print("Average number of created offers in all journeys", sum(counts)/len(counts))

# Investigate cycles in transition system
def print_cycles(g):
    for c in nx.simple_cycles(g):
        count = 0
        c.append(c[0])
        for i in range(len(c)-1):
            count += g[c[i]][c[i+1]]['cost']
        print(len(c))
        print("count", count)

# Compute abbreviations for every event to have shorter names
def compute_abbreviations(log_before, log_after):
    # construct abbreviations for single events

    events_before = set([x['concept:name'] for trace in log_before for x in trace])
    events_after = set([x['concept:name'] for trace in log_after for x in trace])
    events = events_before.union(events_after)

    events_abbreviation = {}
    for e in events:
        numbers = [int(s) for s in e.split(" ") if s.isdigit()]
        if not numbers:
            events_abbreviation[e] = e[:3]
        else:
            if len(numbers) != 1:
                print(e)
                print(numbers)
            assert(len(numbers)==1)
            events_abbreviation[e] = e[:3]+str(numbers[0])
        if "online only" in e:
            events_abbreviation[e] += ".o"
        if "mail and online" in e:
            events_abbreviation[e] += ".m+o"
        if "SHORT" in e:
            events_abbreviation[e] += ".s"
        if "SUPER LONG" in e:
            events_abbreviation[e] += ".sl"
        elif "LONG" in e:
            events_abbreviation[e] += ".s"

    return events_abbreviation

# generate labels s.t. same nodes have same label in both plots
# new_labels shortens the node names and new_labels_int assigns integer codes
def compute_labels(g1, g2, events_abbreviation):
    new_labels = {}
    new_labels_int = {}
    
    # construct union over both nodes
    both_nodes = []#list(set(g1.nodes).union(g2.nodes))
    for n in g1.nodes:
        if n not in both_nodes:
            both_nodes.append(n)
    for n in g2.nodes:
        if n not in both_nodes:
            both_nodes.append(n)
    
    for n,i in zip(both_nodes, range(len(both_nodes))):
        h = [e.strip() for e in n.split("-")]
        h1 = events_abbreviation[h[0]]
        for e in h[1:]:
            h1 += " - "
            h1 += events_abbreviation[e]
        new_labels[n] = h1
        new_labels_int[n] = i

    new_labels.pop("start")
    
    new_labels_int.pop("start")
    return new_labels_int

# Searches for 1:1 connections in graph
def get_connection(g, labels):
    for s in g:
        if len(list(g[s]))!= 1:
            continue
        assert(len(list(g[s]))==1)
        v = list(g[s])[0]

        edges = list(g.in_edges(v))
        if len(edges) == 1:
            s1 = edges[0][0]
            v1 = edges[0][1]
            assert(s == s1 and v == v1)
            if labels[v] != 42: # node with label 42 is not merged - only one that leaves pos (otherwise it seems loke 4 are leaving pos)
                return (s,v)
    return None

# Function to merge all 1:1 edges, produces a smaller plot
def merge_connections(g, labels):
    g = copy.deepcopy(g)
    edge = get_connection(g,labels)
    while(edge != None):
        assert(edge in g.edges)
        print("merged", edge)
        # ensure that start node remains
        g = nx.contracted_nodes(g, edge[0], edge[1])
        g.remove_edges_from(nx.selfloop_edges(g))
        edge = get_connection(g, labels)
        
    return g

# Plots image of reduced decision boundary
def Figure5(g, name, labels, layout = "sfdp", color_path = False, save_elements = False):
    g = copy.deepcopy(g)
    
    nodes_reaching = []
    dec_bound = []

    print("Decision boundary:")
    for s in g.nodes:
        if g.nodes[s]['decision_boundary']:
            g.nodes[s]['color'] = "blue"
            g.nodes[s]['shape'] = "square"
            print(s,labels[s])
            nodes_reaching.extend(g.in_edges(s))
            dec_bound.append(s)
    print("nodes_reaching", len(set(nodes_reaching)), set([n[0] for n in nodes_reaching]))
    print("Number of paths from start to boundary ", len(list(nx.all_simple_paths(g, "start", dec_bound))))
    # colour timeout edges
    for e in g.edges:
        if "TIMEOUT" in g[e[0]][e[1]]['action']:
            g[e[0]][e[1]]['color'] = "#9d4edd"
        elif 'contraction' in g.edges[e]:
            # consider case that timeout edge was ''merged away'' (do overapproximation)
                for e_inner in g.edges[e]['contraction']:
                    if e_inner[0] == e[0]: # make sure that both start in same node
                        if "TIMEOUT" in g.edges[e]['contraction'][e_inner]['action']:
                            g[e[0]][e[1]]['color'] = "#9d4edd"

    edges = nx.bfs_edges(g, "start")
    nodes = ["start"] + [v for u, v in edges]   
    
    #g = nx.relabel_nodes(g, labels) # names needed for correlation computation
    # merge 1:1 connections for nicer plot
    g = merge_connections(g, labels)

    # merge start cluster together
    for n in nx.descendants_at_distance(g, "start", 2).union(nx.descendants_at_distance(g, "start", 1)):
        g = nx.contracted_nodes(g, "start", n)
    
    g.remove_edges_from(nx.selfloop_edges(g))
    
    for s in g:
        outgoing_sum = 0
        if "color" in g.nodes[s]:
            if g.nodes[s]['color'] == "blue":
                    continue
        for n in g[s]:
            if "cost" in g[s][n]:
                outgoing_sum += g[s][n]["cost"]
        if outgoing_sum >= 16:
            g.nodes[s]['color'] = "#0ead69"
        elif outgoing_sum <= -30:
            g.nodes[s]['color'] = "#9a031e"
        else:
            g.nodes[s]['color'] = "grey"

    A = to_agraph(g)
    edge_weights = nx.get_edge_attributes(g,'edge_weight')
    
    # process graph for readability in paper
    for e in g.edges:
        edge = A.get_edge(e[0], e[1])
        if 'controllable' in g[e[0]][e[1]]:
            # check if all edges in merged edge are controllable
            if not g[e[0]][e[1]]['controllable']:
                edge.attr["style"] = "dotted"
            elif 'contraction' in g.edges[e]:
                for e_inner in g.edges[e]['contraction']:
                    if e_inner[0] == e[0]:
                        if not g.edges[e]['contraction'][e_inner]['controllable']:
                            edge.attr["style"] = "dotted"
        edge.attr["label"] = ""
    
    A.add_edge("secret_start", "start")
    for n in A.nodes():
        n.attr['fontsize'] = 100
        n.attr['penwidth'] = 15
        n.attr['height'] = 2
        n.attr['width'] = 2
        if n == "pos":
            n.attr['color'] = "#0ead69"
            n.attr['fontcolor'] = "#0ead69"
            n.attr['label'] = 'Cpos'
        elif n == "neg":
            n.attr['color'] = "#9a031e"
            n.attr['fontcolor'] = "#9a031e"
            n.attr['label'] = 'Cneg'
        elif n not in ["pos", "neg"]:
            if n in labels and labels[n] == 42:
                n.attr['label'] = "s"
            else:
                n.attr['label'] = ""
        if n in shortest_pos_trace_abstracted:
            if color_path:
                n.attr['style'] = 'filled'
        if n == "start":
            n.attr['label'] = "s₀" #"s0"
        if n == "secret_start":
            n.attr['shape'] = "none"
            n.attr['width'] = 1
            n.attr['height'] = 1
    
    for e in A.edges():
        e.attr['penwidth'] = 15
    
    A.graph_attr.update(size="7.75,10.25")
    A.write(name.split(".")[0]+".dot")
    A.layout(layout)
    print("Plotted", name)
    
    if save_elements:
        A.draw(name)
    return g

# computes the after length of paths until timeout event
def pos_distance_unmerged(g):
    count = 0
    length_sum = 0
    length_dict = {}
    for e in g.edges:
        if "TIMEOUT" in g.edges[e]['action']:
            length = nx.shortest_path_length(g, "start", e[0])
            length_sum += length
            count += 1
            if length in length_dict:
                length_dict[length] += 1
            else:
                length_dict[length] = 1
    print(length_dict)
    print("average length until timeout", length_sum/count, "count", count, "length", length_sum)


def color_graph(g, green = 4, red = -4):
    g = copy.deepcopy(g)
    for e in g.edges:
        if g[e[0]][e[1]]['cost'] > green:
            g[e[0]][e[1]]['color'] ="green"
        elif g[e[0]][e[1]]['cost'] < red:
            g[e[0]][e[1]]['color'] ="red"
        else:
            g[e[0]][e[1]]['color'] ="gray"
    
    return g

def draw_dfg(g, name, layout = "sfdp"):
    A = to_agraph(g)
    edge_weights = nx.get_edge_attributes(g,'edge_weight')
    for e in edge_weights:
        e = A.get_edge(e[0], e[1])
        e.attr["fontsize"] = "20"
    for e in g.edges:
        if 'controllable' in g[e[0]][e[1]]:
            if not g[e[0]][e[1]]['controllable']:
                edge = A.get_edge(e[0], e[1])
                edge.attr["style"] = "dotted"

    A.write(name.split(".")[0]+".dot")
    A.layout(layout)
    print("Plotted", name)
    A.draw(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'main_bpi2017',
                        description = "Executes experiments to reproduce results for BPIC2017.")
    parser.add_argument('verifyta_path', help = "UPPAAL Stratego path")
    parser.add_argument('-bpi_path', '-p', help = "BPIC2017 data", default = 'data/orig/BPI Challenge 2017.xes')
    parser.add_argument('-actors', '-a', help = "Actor information", default = 'data/activities2017.xml')
    parser.add_argument('-save_elements', '-save_elements', help = "Set True for storing of plost etc.", default = True, type = bool)

    args = parser.parse_args()

    # Preprocessing of the log
    # Several operations are performed:
    # - The log is split into two parts at the concept drift
    # - Call events ('W_Call incomplete files' and 'W_Call after offers') are accumulated and discretized by time. 
    # - - With a runtime of 60 - 600 sec. is the event considered as "SHORT"
    # - - between 10min and 4h as "LONG"
    # - - exceeding 4h as "SUPER LONG"
    # - Other workflow events are ignored
    # - Cancellation events after A_Pending are ignored
    # - 'O_Create Offer' is tagged (enumerted to consider how many offers were created)
    # - s_0 and terminal states are appended
    print("### Preprocessing ###")
    log_before, log_after = bpic17.preprocessed_log(args.bpi_path)

    print("### Different outcomes ###")
    print("Before log")
    compare_outcomes(log_before)
    print("After log")
    compare_outcomes(log_after)
    print("### Investigate log ###")
    print("Before log")
    investigate_log(log_before)
    print("After log")
    investigate_log(log_after)

    # Process Model Construction
    g_before = game_construction.game(log_before, SEQUENCE_HISTORY, game_construction.sequence, actor_path = args.actors)
    g_after = game_construction.game(log_after, SEQUENCE_HISTORY, game_construction.sequence, actor_path = args.actors)
    g_before = color_graph(g_before)
    g_after = color_graph(g_after)

    # test cycles in transition system
    print("### Before log ###")
    print_cycles(g_before)
    print("")
    print("### After log ###")
    print_cycles(g_after)

    # unrolling of user journey game
    target = [s for s in g_before.nodes if "positive" in s or "negative" in s]
    g_before_unroll = game_operations.unroll(g_before, "start", target, 1)
    target = [s for s in g_after.nodes if "positive" in s or "negative" in s]
    g_after_unroll = game_operations.unroll(g_after, "start", target, 1)

    events_abbreviation = compute_abbreviations(log_before, log_after)
    labels = compute_labels(g_before, g_after, events_abbreviation)

    if args.save_elements:
        uppaal_utils.to_uppaal(g_before, "out/bpi2017_before.xml", layout = "dot")
        uppaal_utils.to_uppaal(g_before_unroll, "out/bpi2017_before_unroll.xml")

        uppaal_utils.to_uppaal(g_after, "out/bpi2017_after.xml", layout = "dot")
        uppaal_utils.to_uppaal(g_after_unroll, "out/bpi2017_after_unroll.xml")

    # construct decision boundary
    # Function to create decision boundary plot, Fig. 4.
    # The decision boundary contains the states where the future outcome is decided.
    # There exists no chance to recover from the decision, the outcome is by then decided.
    db_game_before, db_before = game_operations.compute_db(g_before, "guaranteed_tool.q", uppaal_stratego=args.verifyta_path, output = "./")
    db_game_after, db_after = game_operations.compute_db(g_after, "guaranteed_tool.q", uppaal_stratego=args.verifyta_path, output = "./")

    draw_dfg(db_game_before, "out/graph_weight_before.ps", "dot")
    draw_dfg(db_game_after, "out/graph_weight_after.ps", "dot")

    # apply reduction
    reduced_db_game_before = game_operations.db_reduction(db_game_before)
    reduced_db_game_after = game_operations.db_reduction(db_game_after)

    # to plot positive trace in image
    helper_log = [[t['concept:name'] for t in log_before[i]] for i in range(len(log_before))]
    index = np.argmin([len(t) if "positive" in t else 1000 for t in helper_log])
    shortest_pos_trace = copy.deepcopy(log_before[index])
    shortest_pos_trace_abstracted = [game_construction.sequence(shortest_pos_trace[max(0,pos_index-SEQUENCE_HISTORY+1):pos_index+1]) for pos_index in range(1, len(shortest_pos_trace))]
    shortest_pos_trace_abstracted.insert(0, "start")

    print("Shortest positive trace")
    print(shortest_pos_trace_abstracted)

    # Analyse decision boundary
    print("Label of node reached from C_pos: ", labels['O_Sent (mail and online) - O_Create Offer 1 - O_Sent (mail and online)'])
    printed = Figure5(reduced_db_game_before, "out/clustered_before.png", labels, "dot", color_path=True, save_elements=args.save_elements)
    print("Nodes", len(printed.nodes))
    print("Maximum label in before graph", max([labels[s] for s in reduced_db_game_before if s in labels]))
    print("Leaving positive", len(reduced_db_game_before['pos']))
    for e in reduced_db_game_before['pos']:
        if e in labels:
            print(e, labels[e])
        else:
            print(e)
    print()

    print('O_Sent (mail and online) - O_Create Offer 1 - O_Sent (mail and online)' in reduced_db_game_after)
    printed = Figure5(reduced_db_game_after, "out/clustered_after.png", labels, "dot", save_elements=args.save_elements)
    print("Nodes", len(printed.nodes))
    print("Leaving positive", len(reduced_db_game_after['pos']))
    for e in reduced_db_game_after['pos']:
        if e in labels:
            print(e, labels[e])
        else:
            print(e)

    # compute average length until timeout
    # has to be investigated on original models (with no merged connections)
    print("### Trace lengths ###")
    print("Before")
    pos_distance_unmerged(db_game_before)
    print("After")
    pos_distance_unmerged(db_game_after)

    # Investigate correlations
    print("### Investigate correlations ###")
    # build static decision boundaries
    db_game_before_static, db_before_static = game_operations.compute_db(g_before, "guaranteed_tool.q", uppaal_stratego=args.verifyta_path, output = "./", static=True)
    db_game_after_static, db_after_static = game_operations.compute_db(g_after, "guaranteed_tool.q", uppaal_stratego=args.verifyta_path, output = "./", static=True)
    reduced_db_game_before_static = game_operations.db_reduction(db_game_before_static, static=True)
    reduced_db_game_after_static = game_operations.db_reduction(db_game_after_static, static=True)

    # Reduction detectede include "positive" and "negative" as there are no transitions with the removed feature from a state to positive/negative.
    print("Before")
    bpi_utils.analyse_correlations(log_before, g_before, reduced_db_game_before_static, "out/before_correlations.png", SEQUENCE_HISTORY, game_construction.sequence)
    print("After")
    bpi_utils.analyse_correlations(log_after, g_after, reduced_db_game_after_static, "out/after_correlations.png", SEQUENCE_HISTORY, game_construction.sequence)