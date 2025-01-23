import argparse
import networkx as nx
from src.game_construction import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'build_game',
                        description = "Transforms a directly follows model produces by 'process_model.py' into a game by annotating (un)controllable actions.",)
    parser.add_argument('input', help = "Input model")
    parser.add_argument('output', help = "Output path for game") 
    parser.add_argument('activities', help = "JSON-file storing controllability of edges") 

    args = parser.parse_args()

    g = nx.read_gexf(args.input)

    annotate_actors(g, load_actors(args.activities)) 

    name = args.output + "GAME" + args.input.split("/")[-1].split(".")[0].split("PMODEL")[-1] + "_" + "actors:" + args.activities.split("/")[-1]+'.gexf'
    nx.write_gexf(g, name)
    print("Generated:", name)