import argparse
from pm4py.objects.log.importer.xes import importer as xes_importer
import networkx as nx
import json 
import numpy as np
from src.game_construction import *
#from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'process_model',
                        description = "Takes an processed event log as input and computes a directly-follows process model with weights.",)
    parser.add_argument('input', help = "Input event log") 
    parser.add_argument('output', help = "Output path for process model") 
    parser.add_argument('-t', '--type', help = "Type of directly follows model: default = hist", default = "sequence", choices = ["sequence", "multiset"]) 
    parser.add_argument('-hist', '--history', help = "Number of past steps to be included; default = 3", default = 3, type = int) 

    args = parser.parse_args()

    log = xes_importer.apply(args.input)

    g = model(log, args.history, ms if args.type == "multiset" else sequence)

    name = args.output + "PMODEL" + "_" + "input:"+ args.input.split("/")[-1].split(".")[0]  + "_" + "type:" + args.type + "_"+ "history:"+ str(args.history) + '.gexf'
    # "_" + datetime.today().strftime('%Y-%m-%d#%H:%M:%S')
    # not sure if datetime needed
    nx.write_gexf(g, name) 
    print("Generated:", name)