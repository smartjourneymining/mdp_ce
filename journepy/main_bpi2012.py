import pandas as pd
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import copy
import argparse
# local libraries
from src.preprocessing import bpic12, preprocessing_utils
from src import game_construction, game_operations
import bpi_utils
# global variables
#paths
BPI_PATH_PROCESSED = 'data/processed/bpic12_processed.xes'
# log processing
MIN_SPEAKING_TIME = 60
DAY_DIFFERENCE = 10
CLUSTER_COMPONENTS = 3
# verfification

# Investigate log on average number of successful and unsuccessful journeys and created offers per journey type.
# Results are logged for analysis.
def investigate_log(log_application):
    # investigate avg. number of calls in trace
    counts_positive = {"W_Nabellen incomplete" : 0, "W_Nabellen offertes":0}
    pos_count = 0
    for trace in log_application:
        if preprocessing_utils.contains(trace, "positive") :
            pos_count += 1
            for e in trace:
                if "W_Nabellen incomplete" in e['concept:name']:
                    counts_positive["W_Nabellen incomplete"] += 1
                if "W_Nabellen offertes" in e['concept:name']:
                    counts_positive['W_Nabellen offertes'] += 1

    counts_negative = {"W_Nabellen incomplete" : 0, "W_Nabellen offertes":0}
    neg_count = 0
    for trace in log_application:
        if preprocessing_utils.contains(trace, "negative") :
            neg_count += 1
            for e in trace:
                if "W_Nabellen incomplete" in e['concept:name']:
                    counts_negative["W_Nabellen incomplete"] += 1
                if "W_Nabellen offertes" in e['concept:name']:
                    counts_negative['W_Nabellen offertes'] += 1

    both = {"W_Nabellen incomplete" : 0, "W_Nabellen offertes":0}
    for k in counts_positive:
        both[k] = (counts_negative[k] + counts_positive[k])/(neg_count+pos_count)

    for k in counts_positive:
        counts_positive[k] /= pos_count
    for k in counts_negative:
        counts_negative[k] /= neg_count

    print("Average number of calls in successful journeys", counts_positive)
    print("Average number of calls in unsuccessful journeys", counts_negative)
    print("Average number of calls in all journeys", both)

    # Investigate the average number of created offers
    counts_pos = []
    counts_neg = []
    for trace in log_application:
        count = 0
        if preprocessing_utils.contains(trace, "positive") :
            for e in trace:
                if "O_CREATED" in e['concept:name']:
                    count += 1
            counts_pos.append(count)
        if preprocessing_utils.contains(trace, "negative") :
            for e in trace:
                if "O_CREATED" in e['concept:name']:
                    count += 1
            counts_neg.append(count)
    print("Average number of created offers in successful journeys", sum(counts_pos)/len(counts_pos))
    print("Average number of created offers in unsuccessful journeys", sum(counts_neg)/len(counts_neg))

    counts = []
    for trace in log_application:
        count = 0
        for e in trace:
            if "O_CREATED" in e['concept:name']:
                count += 1
        counts.append(count)
    print("Average number of created offers in all journeys", sum(counts)/len(counts))

# Function compaes both tested refinements - multi-set and sequence - and prints the resulting number of loops and nodes.
def compare(log, actor_path, start = 3, end = 6):
    r = []
    for type in ["multiset", "sequence"]:
        r.append("Refinement "+ type)
        for hist in range(start,end+1):
            # use library tool to build process models
            refinement = game_construction.ms if type == "multiset" else game_construction.sequence
            inner_game = game_construction.game(log, hist, refinement, actor_path=actor_path)
            r.append("History: "+ str(hist)+ " Loops: "+str(len(list(nx.simple_cycles(inner_game)))) + " Nodes: " +str(len(inner_game.nodes())))
    return r

# function to retreive decision boundary states from game
def get_dec_bound(g):
    bound = []
    for s in g.nodes:
        if "decision_boundary" in g.nodes[s]:
            if g.nodes[s]["decision_boundary"]:
                bound.append(s)
                print(s)
    return bound

# Prints elements of decision boundary
def analyse_decision_boundary(bound):
    print("States in decision boundary", len(bound))
    # bound contains no element with no call ("W_Nabellen")
    for n in bound:
        if "W_Nabellen" not in n:
            print(n)

    print("States with one cancellation")
    for n in bound:
        if '"O_CANCELLED": 1' in n:
            print(n)
    print()
    print("States with two cancellations")
    for n in bound:
        if '"O_CANCELLED": 2' in n:
            print(n)
    print()
    print("States with no cancellations")
    for n in bound:
        if "O_CANCELLED" not in n:
            print(n)

######################
##### Plotting #######
######################

# Helper function to draw plot.
# Mainly used in analysis phase.
def draw_dfg(g, name, layout = "sfdp"):
    A = to_agraph(g)
    edge_weights = nx.get_edge_attributes(g,'cost')
    for e in edge_weights:
        e = A.get_edge(e[0], e[1])
        #e.attr["penwidth"] = edge_weights[e]*scaling
        e.attr["fontsize"] = "20"
        e.attr["label"] = edge_weights[e]
    for e in g.edges:
        if 'controllable' in g[e[0]][e[1]]:
            if not g[e[0]][e[1]]['controllable']:
                edge = A.get_edge(e[0], e[1])
                edge.attr["style"] = "dotted"

    for s in g.nodes:
        if "decision_boundary" in g.nodes[s]:
            if g.nodes[s]["decision_boundary"]:
                g.nodes[s]['color'] = "blue"
    A.write(name.split(".")[0]+".dot")
    A.layout(layout)
    print("Plotted", name)
    A.draw(name)

# Produces Figure 6 in the publication - reduced decision boundary plot
# Highlights decision boundary as blue squares, edges are colored black, red, green or gray.
# Reducible nodes are merged into resp. Cneg or Cpos.
def Figure_6(g, name, layout = "sfdp"):
    # color graph
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
    edge_weights = nx.get_edge_attributes(g,'cost')
    scaling = 10

    for e in edge_weights:
        e = A.get_edge(e[0], e[1])
        e.attr["penwidth"] = "15"
        e.attr["fontsize"] = "20"
        e.attr["label"] = ""
        e.attr["color"] = "red" if edge_weights[e] < 0 else "green"
        if e[1] == 'neg':
            e.attr["color"] = "gray"
    for e in g.edges:
        if 'controllable' in g[e[0]][e[1]]:
            if not g[e[0]][e[1]]['controllable']:
                edge = A.get_edge(e[0], e[1])
                edge.attr["style"] = "dotted"
            elif 'contraction' in g.edges[e]:
                for e_inner in g.edges[e]['contraction']:
                    if e_inner[0] == e[0]:
                        if not g.edges[e]['contraction'][e_inner]['controllable']:
                            edge = A.get_edge(e[0], e[1])
                            edge.attr["style"] = "dotted"
    clusters = {"0": [], "1": [], "2":[]}
    for s in A.nodes():
        if "decision_boundary" in s.attr:
            if s.attr["decision_boundary"] == "True":
                s.attr['color'] = "blue"
                
                c = "0" if not "O_CANCELLED" in s else "1" if '"O_CANCELLED": 1' in s else "2"
                clusters[c].append(s)

    A.add_edge("secret_start", '{"A_SUBMITTED": 1, "start": 1}') # add to draw start edge
    e = A.get_edge("secret_start", '{"A_SUBMITTED": 1, "start": 1}')
    e.attr["penwidth"] = "20"
    e.attr["fontsize"] = "50"
    e.attr["label"] = ""
    for n in A.nodes():
        n.attr['fontsize'] = 100
        n.attr['penwidth'] = 20
        n.attr['height'] = 2
        n.attr['width'] = 2
        if n == "pos":
            n.attr['color'] = "#0ead69"
            n.attr['fontcolor'] = "#0ead69"
            n.attr['label'] = 'Cpos'
        if n == "neg":
            n.attr['color'] = "#9a031e"
            n.attr['fontcolor'] = "#9a031e"
            n.attr['label'] = 'Cneg'
        
        if n not in ["pos", "neg"]:
            n.attr['label'] = ""
        if n == '{"A_SUBMITTED": 1, "start": 1}': # merged edges remove "start"
            n.attr['label'] = "s₀" #"s0"
        if n == "secret_start":
            n.attr['shape'] = "none"
    
    A.layout(layout)
    print("Plotted", name)
    A.draw(name)

# Searches for 1:1 connections in graph
def get_connection(g):
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
            return (s,v)
    return None

# Function to merge all 1:1 edges, produces a smaller plot
def merge_connections(g):
    g = copy.deepcopy(g)
    edge = get_connection(g)
    while(edge != None):
        g = nx.contracted_nodes(g, edge[1], edge[0])
        g.remove_edges_from(nx.selfloop_edges(g))
        edge = get_connection(g)
        
    return g

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'main_bpi2012',
                        description = "Executes experiments to reprduce results for BPIC2012.")
    parser.add_argument('verifyta_path', help = "UPPAAL Stratego path")
    parser.add_argument('-bpi_path', '-p', help = "BPIC2012 data", default = 'data/orig/BPI_Challenge_2012.xes')
    parser.add_argument('-actors', '-a', help = "Actor information", default = 'data/activities2012.xml')
    parser.add_argument('-save_elements', '-save_elements', help = "Set True for storing of plost etc.", default = True, type = bool)

    args = parser.parse_args()

    # # Preprocessing
    # In the preprocessing, we:
    # - transform the log into a list format
    # - discretise offer events
    # - remove trivial events
    # - remove incomplete and declines journeys
    # - compute durations of calls
    # - cluster call events 
    print("### Preprocessing ###")
    log_application = bpic12.preprocessed_log(args.bpi_path)

    print("### Investigate log ###")# # Log Properties
    # We investigate certain log properties differentiating the BPIC'17 event log from BPIC'12.
    # It shows that not only the average number of calls but also the average number of generated offers were significantly changed.
    investigate_log(log_application)

    # # Process Model Construction
    # The process model is build with the published decision boundary tool.
    # We iterate over different settings to find the most promising one.
    # The final model uses the muli-set refinement with a history of 5.
    print("### Comparing different model parameters ###")
    for e in compare(log_application, args.actors):
        print(e)

    # # Decision Boundary Analysis
    # The decision boundary of the process model with the multi-set refinement and history of 5 is analysed.
    # In the accompanying paper we compare the properties of the BPIC'12 decision boundary with the ones from BPIC'17.

    #  Build process model & user journey game
    g = game_construction.game(log_application, 5, game_construction.ms, actor_path = args.actors)

    # plot process model for analysis
    if args.save_elements:
        draw_dfg(g, "out/bpi2012.ps", "dot")

    # construct decision boundary
    db_game, db = game_operations.compute_db(g, "guaranteed_tool.q", uppaal_stratego=args.verifyta_path, output = "./")

    # apply reduction
    reduced_db_game = game_operations.db_reduction(db_game)

    print("### Investigate decision boundary ###")
    bound = get_dec_bound(reduced_db_game)
    analyse_decision_boundary(bound)
    reduced_db_game = merge_connections(copy.deepcopy(reduced_db_game)) # called with deepcopy to ensure that original graph is not changes - correlation analysis later

    print("Leaving positive", len(reduced_db_game['pos']))
    for e in reduced_db_game.out_edges("pos"):
        print(e[1], reduced_db_game.edges[e])

    if args.save_elements:        
        Figure_6(reduced_db_game, "out/clustered_2012.png", layout = "dot")

    # # Dimension Reduction
    # We investigate the applicability of the decision boundary as dimension reduction technique.
    # When interpreting single events as features, many events are highly correlated.
    # For instance, when entering the positive or negative cluster the outcome is determined and many events are thus positively correlated.
    # When removing the events contained in the decision boundary, the present correlations reduce drastically.

    print("### Analyse correlation reduction ###")
    # construct static decision boundary
    static_db_game, static_db = game_operations.compute_db(g, "guaranteed_tool.q", uppaal_stratego=args.verifyta_path, output = "./", static=True)
    # apply reduction
    reduced_static_db_game = game_operations.db_reduction(static_db_game, static=True)

    bpi_utils.analyse_correlations(log_application, g, reduced_static_db_game, "out/2012_correlations.png", 5, game_construction.ms, save_elements=args.save_elements)

"""
# allows to plot the decision boundary as independent graph and analyse the cancellation clusters; requires that input models are in multi-set form
# (not part of publication)

def to_set(bound):
    bounds = []
    for n in bound:
        b = {}
        n = n.replace("{","")
        n = n.replace("}","")
        n = n.replace('"',"")
        n = n.replace(' ',"")
        s = n.split(",")
        for e in s:
            s1 = e.split(":")
            b[s1[0]] = int(s1[-1])
        bounds.append(b)
    return bounds
bounds = to_set(bound)

def dist(s,t):
    not_contained = 0
    for k in s:
        if k not in t:
            not_contained += 1
        elif s[k] != t[k]:
            not_contained += abs(s[k]-t[k])
    return not_contained

def build_dec_bound_graph(bounds):
    g = nx.Graph()
    for s in bounds:
        g.add_node(str(s))
    for s in bounds:
        for t in bounds:
            if s == t:
                continue
            if dist(s,t) <= 1:
                g.add_edge(str(s),str(t))
    return g
g = build_dec_bound_graph(bounds)

def cluster_contains(c, s):
    for e in c:
        if s not in e:
            return False
    return True

def unpublished_analyse_decision_boundary(file_name, name):
    g_dec_boundary = nx.read_gexf(file_name)
    bound = get_dec_bound(g_dec_boundary)
    bounds = to_set(bound)
    g = build_dec_bound_graph(bounds)

    A = to_agraph(g)

    count = 0
    for c in nx.connected_components(g):
        print(list(c)[0])
        print(len(c))

        label = "0" if not cluster_contains(c,"O_CANCELLED") else "1" if cluster_contains(c,"'O_CANCELLED': 1") else "2"
        
        if label == "0":
            assert(not cluster_contains(c,"'O_CANCELLED': 1"))
            assert(not cluster_contains(c,"'O_CANCELLED': 2"))

        if label == "1":
            assert(not cluster_contains(c,"'O_CANCELLED': 2"))

        if label == "2":
            assert(not cluster_contains(c,"'O_CANCELLED': 1"))

        colors = ["blue", "red", "orange"]
        A.add_subgraph(c, name='cluster_'+str(count), label= "#"+label, color = colors[int(label)], fontsize = 60, fontcolor = colors[int(label)])

        count+=1

    for n in A.nodes():
        n.attr['penwidth'] = 5
        n.attr["label"] = ""

    A.layout("dot")
    print("Plotted", name)
    A.draw(name)
    
    nx.draw(g, node_size=80)

"""