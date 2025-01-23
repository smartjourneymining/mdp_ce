from src.preprocessing import greps, preprocessing_utils
from src import game_construction, game_operations
from src.mc_utils import uppaal_utils
import argparse
import networkx as nx
import subprocess
from networkx.drawing.nx_agraph import to_agraph
import json

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


def write_model_to_csv(g, out):
    """ Writes game as csv file to out

    Args:
        g (Networkx DiGraph): Game to write
        out (str): Path to folder to write strategies to
    """
    with open(out, "w+") as f:
        f.write("start" + ',' + "target" + ',' + "action" + ',' + "controllable" + ',' + "gas" + '\n')
        for line in g.edges(data=True): # prints every edge with more information
            print(line)
            print("parsed", str(line[0]) + ',' + str(line[1]) + ',' + str(line[2]['action'])+ ',' + str(line[2]['controllable']) + ',' + str(line[2]['cost']))
            f.writelines(str(line[0]) + ',' + str(line[1]) + ',' + str(line[2]['action'])+ ',' + str(line[2]['controllable']) + ',' + str(line[2]['cost']) + '\n')

def original_state(state, original_nodes):
    """Compute the original state name from a state generated in unrolling.
        Unrolling iterates over states by added "vv" between name and number and certain characters are removed from state name.

    Args:
        state (str): State crated in unrolling
        original_nodes (Dict str -> str): Dict mapping Uppaal state names to game state names

    Returns:
        str: Original state in game
    """
    if "vv" in state:
        state = state.split("vv")
        state = original_nodes[state[0]]
    else:
        state = original_nodes[state]
    return state 

def generate_strategies(verifyta_path, g, out_path):
    """Writes generates strategies from g to out_path. Generates strategy file, parses a non-deterministic and refined strategy.

    Args:
        verifyta_path (str): Path to local verifyta file
        g (Networkx DiGraph): Game for which strategies are searched
        out_path (str): Path to FOLDER to write strategies to
    """
    # generate strategy
    with open(out_path+"strategy.q", 'w+') as outfile:
        outfile.write('strategy goPos = control: A<> reached_positive \n')
        outfile.write('saveStrategy("out/goPos.xml", goPos)\n')
        outfile.write('strategy goPosFast = minE(steps) [t<=100] : <> (reached_positive) under goPos\n')
        outfile.write('saveStrategy("out/goPosFast.json", goPosFast)')

    out = subprocess.Popen([verifyta_path, out_path+"unrolled_graph_transition.xml", out_path+"strategy.q"], stdout=subprocess.PIPE, stderr = subprocess.PIPE)
    out.wait()
    results, err = out.communicate()
    results = results.decode("utf-8")
    err = err.decode("utf-8")
    print(err)
    assert(err == "")
    # parse strategy
    original_nodes = {n.replace(":", "").replace(" ","").replace(".", "").replace(",", "").replace("-","") : n for n in g.nodes}

    with open(out_path+'goPos.xml', 'r') as file:
        data=file.read()
    
    non_det_strategy = {}
    data = data.split('State') # split into states
    data = [data[i] for i in range(len(data)) if "you are in" in data[i]] # remove beginning info
    for d in data:
        lines = d.split('\n')
        assert(len(lines)>= 2)
        state = lines[0].split("( ")[1].split(" )")[0].replace("Journey.", "")

        state = original_state(state, original_nodes)

        possible_actions = []
        for s in lines[1:]:
            if not s:
                continue # for empty line
            # parse transition
            assert("When you are in (x<=2)" in s or "While you are in	true" in s or "While you are in	(x<=2), wait." in s)
            if "transition" in s:
                next_state = s.split("->")[1].split(" {")[0].replace("Journey.", "")
                next_state = original_state(next_state, original_nodes)
                action = g.edges[(state, next_state)]['action']
            else:
                assert("wait" in s) # do nothing
                action = "wait"
            possible_actions.append(action)
        if state not in non_det_strategy:
            non_det_strategy[state] = possible_actions
        else:
            # make sure that all copied states have same actions
            for n in possible_actions:
                assert(n in non_det_strategy[state])
    
    # write non_det_strategy
    with open(out_path+"non_det_strategy.csv", "w+") as f:
        f.write("state" + ","+ "action" + '\n')
        for key, value in non_det_strategy.items():
            for v in value:
                #f.writelines(key + "," + ";".join(value)+ '\n')
                f.writelines(key + "," + v + '\n')

    # parse refined strategy
    with open(out_path+'goPosFast.json', 'r') as file:
        data=file.read()
    strat = json.loads(data)
    locations = strat['locationnames']['Journey.location']
    actions = strat['actions']
    regressors = strat['regressors']

    
    strategy_mapping = {}
    for r in regressors:
        best_action = -1
        best_cost = 10000
        for r_help in regressors[r]['regressor']:
            if regressors[r]['regressor'][r_help] < best_cost:
                best_cost = regressors[r]['regressor'][r_help]
                best_action = r_help
        action_start, action_target = actions[best_action].split(" {")[0].replace("Journey.", "").split("->")
        if "vv" in action_start:
            action_start = action_start.split("vv")
            action_start = original_nodes[action_start[0]]
        else:
            action_start = original_nodes[action_start]
        if "vv" in action_target:
            action_target = action_target.split("vv")
            action_target = original_nodes[action_target[0]]
        else:
            action_target = original_nodes[action_target]

        assert(action_start in g.nodes and action_target in g.nodes)

        # add to strategy and assert that all copies have same target
        action = g.edges[(action_start, action_target)]['action']
        if action_start not in strategy_mapping:
            strategy_mapping[action_start] = action
        else:
            print("contained", action_target, strategy_mapping[action_start])
            assert(action == strategy_mapping[action_start])

    refined_strategy = {}
    # refined strategy is subset of strategy
    for s in strategy_mapping:
        assert(s in non_det_strategy)
    for s in non_det_strategy:
        if s not in strategy_mapping:
            # might have more than 1 element, e.g. when all actions same cost
            refined_strategy[s] = non_det_strategy[s][0]
        else:
            refined_strategy[s] = strategy_mapping[s]

    with open(out_path+"refined_strategy.csv", "w+") as f:
        f.write("state" + ","+ "action" + '\n')
        for key, value in refined_strategy.items():
            f.writelines(key + "," + value+ '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'main_bpi2017',
                        description = "Executes experiments to reproduce results for BPIC2017.")
    parser.add_argument('verifyta_path', help = "UPPAAL Stratego path")
    parser.add_argument('-greps_path', '-p', help = "Greps data", default = 'data/orig/data.csv')
    parser.add_argument('-actors', '-a', help = "Actor information", default = 'data/activities.xml')

    args = parser.parse_args()

    print("### Preprocessing ###")
    log = greps.preprocessed_log(args.greps_path)

    print("### Write event log ###")
    preprocessing_utils.export(log, "out/first_sublog.xes")

    g = game_construction.game(log, 2, game_construction.sequence, actor_path = args.actors)
    print(len(g.nodes), len(g.edges), len(list(nx.simple_cycles(g)))) 
    
    """
    #TODO 3 dges are missing -> in developer 84 exists "subjectAcceptanceReceived" and "Results automatically shared" concurrently, are sorted differently 
    g_read = nx.read_gexf("out/helper_graph.gexf")

    print(g_read.nodes)

    for n in g_read.nodes:
        name = n.replace("finPos", "positive")
        name = name.replace("finNeg", "negative")
        if name not in g.nodes:
            print("not contained node", name)

    for e in g_read.edges:
        name1 = e[0].replace("finPos", "positive")
        name1 = name1.replace("finNeg", "negative")
        name2 = e[1].replace("finPos", "positive")
        name2 = name2.replace("finNeg", "negative")
        if (name1,name2) not in g.edges:
            print("not contained", (name1, name2))

    print("#####")
    # new constructed graph  is proper subset
    for n in g.nodes:
        name = n.replace("positive", "finPos")
        name = name.replace("negative", "finNeg")
        assert(name in g_read.nodes)
        if name not in g_read.nodes:
            print("not contained node", name)

    for e in g.edges:
        name1 = e[0].replace("positive", "finPos")
        name1 = name1.replace("negative", "finNeg")
        name2 = e[1].replace("positive", "finPos")
        name2 = name2.replace("negative", "finNeg")
        assert((name1, name2) in g_read.edges)
        if (name1,name2) not in g_read.edges:
            print("not contained", (name1, name2))

    draw_dfg(g, "out/graph.ps", layout="dot")
    draw_dfg(g_read, "out/graph_read.ps", layout="dot")
    assert(False)
    """

    print("### Uppaal Stratego Results ###")
    target = [s for s in g.nodes if "finPos" in s or "finNeg" in s]
    help_g_unroll = game_operations.unroll(g, "start", target, 1)
    uppaal_utils.to_uppaal(help_g_unroll, "out/unrolled_graph_transition.xml", layout = "dot")    

    out = subprocess.Popen([args.verifyta_path, "out/unrolled_graph_transition.xml", "out/unrolled_graph_transition.q"], stdout=subprocess.PIPE, stderr = subprocess.PIPE)
    out.wait()
    results, err = out.communicate()
    results = results.decode("utf-8") 
    err = str(err.decode("utf-8"))
    if err != "":
        results = [-1,1,-1,-1,1,-1]
    else:
        results = results.split("\n")
        results = [results[i+1] for i in range(len(results)-1) if "Formula" in results[i] and "E" in results[i+1]]
        results = [r.replace("≈", "") for r in results]
        results = [float(r.split("±")[0].split("runs)")[1].split("=")[1].strip()) for r in results]
    assert(len(results) == 6)
    print(results)

    print("### Write model to csv ###")
    write_model_to_csv(g, "out/list.csv")

    print("### Generate strategies ###")
    generate_strategies(args.verifyta_path, g, "out/")