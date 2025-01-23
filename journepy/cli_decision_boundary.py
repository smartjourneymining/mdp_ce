import argparse
import networkx as nx
from src.game_operations import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'decision_boundary',
                        description = "Computes the decision boundary of the given game.",)
    parser.add_argument('input', help = "Input model")
    parser.add_argument('output', help = "Output path for game with annotated decision boundary")
    parser.add_argument('uppaal_stratego', help = "Path to Uppaal Stratego's VERIFYTA") 
    parser.add_argument('-d', '--debug', help = "Print additional information", default = False, type = bool) 
    parser.add_argument('-q', '--query', help = "Path to the boolean query for the decision boundary.", default = 'guaranteed_tool.q')
    parser.add_argument('-k', '--unrolling_factor', help = "Constant factor how often every lop is unrolled; default = 0", type = int, default = 0) 
    parser.add_argument('-s', '--static', help = "Game decision boundary with neglecting game properties (static decision boundary); default = False", default = False, type = bool) 

    args = parser.parse_args()

    # Load graph
    g = nx.read_gexf(args.input)

    # Compute single results
    g, results = query(g, args.query)

    # Compute decision boundary
    if not args.static:
        g, db = game_db(g, results)
    else:
        #g, db = db_shortcut(g)
        g = reachable_cluster(g, results)

    name = args.output+"DECB"+ args.input.split("/")[-1].split(".")[0].split("GAME")[-1] + "_unrolling_factor:" + str(args.unrolling_factor) + "_" + ".gexf"

    nx.write_gexf(g, name)
    print("Generated:", name)
