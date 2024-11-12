from journepy.src.preprocessing.greps import preprocessed_log
from journepy.src.alergia_utils import convert_utils
# from journepy.src.mc_utils.prism_utils import PrismPrinter
# from journepy.src.mc_utils.prism_utils import PrismQuery

# import probabilistic_game_utils as pgu 

from aalpy.learning_algs import run_Alergia
from aalpy.utils import save_automaton_to_file
from IPython.display import Image


import pandas as pd
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import to_agraph

import json

import networkx as nx

import subprocess

import matplotlib.pyplot as plt

import os

import gurobipy as gp
from gurobipy import GRB, norm

import random
random.seed(42)

import pickle
REGENERATE = False
import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(
path=path, # path to the root directory
project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
dotenv=True, # load environment variables from .env if exists in root directory
pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
cwd=True, # change current working directory to the root directory (helps with filepaths)
)

def build_benchmarks():
    # load actor mapping: maps events to an actor (service provider or user)
    print(os.getcwd())
    with open('data/activities_greps.xml') as f:
        data = f.read()
    actors = json.loads(data)
    
    
    # build action mapping: assigns each event to an actor
    actions_to_activities = {}
    for a in actors:
        if actors[a] == "company":
            # if a in ['vpcAssignInstance', 'Give feedback 0', 'Results automatically shared', 'waitingForActivityReport']: # todo: might be quite realistic?
            #     actions_to_activities[a] = "company"
            # else:  
            #     actions_to_activities[a] = a
            actions_to_activities[a] = 'company'
        else:
            # if a == "negative":
            #     actions_to_activities[a] = "user"
            # elif "Give feedback" in a or "Task event" in a:
            #     actions_to_activities[a] = a
            # else:
            #     actions_to_activities[a] = "user"
            actions_to_activities[a] = a
    
    filtered_log = preprocessed_log("data/data.csv", include_loggin=False) # also discards task-event log-in   
    
    # change from xes format
    filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log]
    
    # print(filtered_log_activities[0])
    # for t in filtered_log_activities:
    #     new_insert = []
    #     for i in range(len(t)):
    #         if "Give feedback" in t[i]:
    #             new_insert.insert(0, (i, "Work "+t[i])) # fix insert
    #     for i in new_insert:
    #         t.insert(i[0], i[1])
    # print(filtered_log_activities[0])
    
    data = [[(actions_to_activities[t[i]], t[i]) for i in range(1, len(t))] for t in filtered_log_activities]
    for d in data:
        d.insert(0, 'start')
        
    
    # quantify environment - distribution of players after for events is learned
    data_environment = []
    for trace in data:
        current = [trace[0]]
        for i in range(1, len(trace)):
            e = trace[i]
            previous_state = "start" if i == 1 else trace[i-1][1]
            
            # encode decision in one step
            current.append(('env', actors[e[1]] + previous_state))
            current.append(e)
        data_environment.append(current)
    
    model = run_Alergia(data_environment, automaton_type='mdp', eps=0.1, print_info=True)
    save_automaton_to_file(model, "out/model.png", file_type="png")
    
    model = convert_utils.mdp_to_nx(model, actors)
    
    return model
    
    
    
def construct_strategy_from_solution(model : nx.DiGraph, p_sa : dict):
    strategy = {}
    for s in model.nodes:
        if "positive" in s or "negative" in s:
            continue
        if model.edges[list(model.edges(s))[0]]['controllable'] or model.edges[list(model.edges(s))[0]]['action'] == 'env':
            continue
        enabled_actions = list(set([model.edges[e]['action'] for e in model.edges(s)]))
        strategy[s] = {a : p_sa[s][a].X for a in enabled_actions}
    
    return strategy
    
    
    
def quadratic_program(model : nx.DiGraph, target_prob : float, user_strategy : dict):
    m = gp.Model("qp")
    p = {s : m.addVar(ub=1.0, name=s, lb = 0) for s in model.nodes}

    p_sa = {}
    
    # encode scheduler constraints
    for s in model.nodes:
        if 'positive' in s:
            m.addConstr(p[s] == 0)
            continue
        if 'negative' in s:
            m.addConstr(p[s] == 1)
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        if 'customer' not in s:
            assert len(enabled_actions) <= 1, "More than one action for non-user state %s" % s 
        p_sa[s] = {a : m.addVar(ub=1.0, name=s+'_'+a, lb = 0) for a in enabled_actions}
        m.addConstr(sum(list(p_sa[s].values())) == 1) # scheduler sums up to one
        for a in enabled_actions:
            m.addConstr(p_sa[s][a] <= 1)
            
    # encode model
    for s in p_sa:
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
        m.addConstr(p[s] == sum([p_sa[s][model.edges[e]['action']] * float(model.edges[e]['prob_weight']) * p[e[1]] for e in model.edges(s)]))
        
    # encode reachability constraint
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    m.addConstr(target_prob >= p[start_state])
    
    
    d_0 = m.addVar(name='d0', lb = 0)
    d_1 = m.addVar(name='d1', lb = 0, ub=1)
    d_inf = m.addVar(name='d_inf', lb = 0, ub=1)

    # strict proximal
    def add_abs(var, prob, constr):
        m.addConstr(prob - constr <= var)
        m.addConstr(constr - prob <= var)
        
    # for s in p_sa:
    #     if model.edges[list(model.edges(s))[0]]['controllable'] or model.edges[list(model.edges(s))[0]]['action'] == 'env':
    #         continue
    #     assert s in user_strategy, f'{s} not in user_strategy'
    #     for a in p_sa[s]:
    #         assert a in user_strategy[s], f'{a} not in {user_strategy[s]}'
    #         add_abs(d_inf, p_sa[s][a], user_strategy[s][a])

    d_sa = {}
    for s in model.nodes:
        if 'positive' in s or 'negative' in s:
            continue
        if model.edges[list(model.edges(s))[0]]['controllable'] or model.edges[list(model.edges(s))[0]]['action'] == 'env':
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        if 'customer' not in s:
            assert len(enabled_actions) == 1, f'More than one action for non-user state {s}' 
        d_sa[s] = {a : m.addVar(ub=1.0, name=f'Abs dist state {s} action {a}', lb = 0) for a in enabled_actions}
        for a in enabled_actions:
            add_abs(d_sa[s][a], p_sa[s][a], user_strategy[s][a])
        # encode d_inf constraint
        m.addConstr(0.5 * sum(d_sa[s].values()) <= d_inf) 
            
    # relaxed proximal
    # use d_sa to encode d_1 norm
    m.addConstr(d_1 == sum([0.5 * sum(d_sa[s].values()) for s in d_sa]) / len(user_strategy))
    #m.addConstr(d_1 == norm([0.5 * sum(d_sa[s].values()) for s in d_sa], 1.0))
    
    # encode sparsity
    decision_changed = {}
    dist_binary = {}
    for s in d_sa:
        dist_binary[s] = m.addVar(ub=1.0, name=f'State {s} was changed', lb = 0, vtype=gp.GRB.BINARY)
        decision_changed[s] = m.addVar(ub=1.0, name=f'Var dist state {s}', lb = 0) 
        m.addConstr(decision_changed[s] == 0.5 * sum(d_sa[s].values()))
        m.addConstr(decision_changed[s] <= 10 * dist_binary[s])
    m.addConstr(sum(dist_binary.values()) == d_0)
    
    m.write("out/gurobi.lp")
    
    m.setObjective(d_0 + d_1 + d_inf, sense = GRB.MINIMIZE)
    m.optimize()
    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")
    print(f"Obj: {m.ObjVal:g}")
    
    print('Constructed solution')
    print(construct_strategy_from_solution(model, p_sa))
    
    strategy_diff(user_strategy, construct_strategy_from_solution(model, p_sa))
    
    return m

def construct_user_strategy(model : nx.DiGraph):
    user_strategy = {}
    
    for s in model.nodes:
        if "positive" in s or "negative" in s:
            continue
        if not model.edges[list(model.edges(s))[0]]['controllable'] and model.edges[list(model.edges(s))[0]]['action'] != 'env':
            enabled_actions = list(set([model.edges[e]['action'] for e in model.edges(s)]))
            distr = random.sample(range(1, 1000), len(enabled_actions))
            user_strategy[s] = {enabled_actions[i] : distr[i] / sum(distr) for i in range(len(enabled_actions))}
    
    return user_strategy

def strategy_diff(strat1 : dict, strat2 : dict):
    assert strat1.keys() == strat2.keys()
    for s in strat1:
        assert s in strat2
        assert strat1[s].keys() == strat2[s].keys()
        for a in strat1[s]:
            if strat1[s][a] != strat2[s][a]:
                print(f'In state {s} action {a} differs, {strat1[s][a]} != {strat2[s][a]}')

if __name__ == '__main__':
    model = build_benchmarks()
    if REGENERATE:
        user_strategy = construct_user_strategy(model) # NOTE strategies are not guaranteed to be equivalent as the set construction can change
        with open('out/user_strategy.pickle', 'wb+') as handle:
            pickle.dump(user_strategy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('out/user_strategy.pickle', 'rb') as handle:
        user_strategy = pickle.load(handle)
    qp = quadratic_program(model, 0.5, user_strategy)
    
    
# TODO write tool to visualize changes - indicate red transitions in model
# TODO test for bpic