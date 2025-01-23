import argparse
import networkx as nx
from src.game_operations import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'decision_boundary_reduction',
                        description = "Merges determined states together to results in one guaranteed positive outcome state and one guaranteed negative outcome state.",)
    parser.add_argument('input', help = "Input model")
    parser.add_argument('output', help = "Output path for reduced game") 
    parser.add_argument('-s', '--static', help = "Game decision boundary with neglecting game properties (static decision boundary); default = False", type = bool, default = False)
    args = parser.parse_args()

    # Load graph
    g = nx.read_gexf(args.input)

    #g = reduce_graph(g)
    g = db_reduction(g, args.static)

    # add graph attributes
    g.graph["reduced_graph"] = True

    name = args.output + args.input.split("/")[-1].split(".")[0] + "reduced:True"+ ".gexf"
    nx.write_gexf(g, name)
    print("Generated:", name)