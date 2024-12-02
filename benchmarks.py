from journepy.src.preprocessing.greps import preprocessed_log as preprocessed_log_greps
from journepy.src.preprocessing.bpic12 import preprocessed_log as preprocessed_log_bpic_2012
from journepy.src.alergia_utils import convert_utils
# from journepy.src.mc_utils.prism_utils import PrismPrinter
# from journepy.src.mc_utils.prism_utils import PrismQuery

# import probabilistic_game_utils as pgu 

from aalpy.learning_algs import run_Alergia
from aalpy.utils import save_automaton_to_file
from IPython.display import Image


import pandas as pd

from networkx.drawing.nx_agraph import to_agraph

import json

import networkx as nx

import subprocess 
import multiprocessing

import matplotlib.pyplot as plt

import os

import gurobipy as gp
from gurobipy import GRB

import cvxpy as cp

import random
random.seed(42)

from Result import Result

import argparse

import pickle
REGENERATE = False
STRATEGY_SLACK = 0.1 

import LogParser

import itertools
from pathlib import Path

import mosek
from z3 import *

import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(
path=path, # path to the root directory
project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
dotenv=True, # load environment variables from .env if exists in root directory
pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
cwd=True, # change current working directory to the root directory (helps with filepaths)
)


def build_benchmark_greps():
    # load actor mapping: maps events to an actor (service provider or user)
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
    
    filtered_log = preprocessed_log_greps("data/data.csv", include_loggin=False) # also discards task-event log-in
    
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
    
    
def build_benchmark_bpic_12():
    from LogParser import BPIC12Parser
    parser = BPIC12Parser('data/BPI_Challenge_2012.xes', 'data/activities_2012.xml')
    return parser.build_benchmark()
    with open('data/activities2012.xml') as f:
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
            
    filtered_log = preprocessed_log_bpic_2012("data/BPI_Challenge_2012.xes") # uses common preprocessing
    
    # change from xes format
    filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log]
    
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
    
    return(False)
    return model

    
def build_benchmark_bpic_17():
    from LogParser import BPIC17BeforeParser, BPIC17AfterParser, BPIC17BothParser
    parser = BPIC17BeforeParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
    return parser.build_benchmark()

def build_benchmark_spotify():
    from LogParser import SpotifyParser
    parser = SpotifyParser('data/spotify/', 'data/activities_spotify.xml')
    return parser.build_benchmark()

def construct_optimal_strategy_from_solution(model : nx.DiGraph, p : dict):
    strategy = {}
    for s in model.nodes:
        if "positive" in s or "negative" in s:
            continue
        if model.edges[list(model.edges(s))[0]]['controllable'] or model.edges[list(model.edges(s))[0]]['action'] == 'env':
            continue
        enabled_actions = list(set([model.edges[e]['action'] for e in model.edges(s)]))
        reward_action = {a : sum([float(model.edges[e]['prob_weight']) * p[e[1]].X for e in model.edges(s) if model.edges[e]['action'] == a]) for a in enabled_actions} # linear combination of possible outcomes for action
        number_optimal_actions = len([a for a in reward_action if reward_action[a] == p[s].X])
        strategy[s] = {a : 1 / number_optimal_actions if reward_action[a] == p[s].X else 0 for a in enabled_actions}
    
    return strategy

def gurobi_access(v):
    return v.X

def mosek_access(v):
    return v.value

def construct_strategy_from_solution(model : nx.DiGraph, p_sa : dict, access, user_strategy = {}):
    strategy = {}
    for s in model.nodes:
        if "positive" in s or "negative" in s:
            continue
        if model.edges[list(model.edges(s))[0]]['controllable'] or model.edges[list(model.edges(s))[0]]['action'] == 'env':
            continue
        enabled_actions = list(set([model.edges[e]['action'] for e in model.edges(s)]))
        if s not in p_sa: # geometric program ignores states wit p = 0
            assert user_strategy
            strategy[s] = user_strategy[s]
        else:
            strategy[s] = {a : access(p_sa[s][a]) for a in enabled_actions}
    
    return strategy
    
    
def minimum_reachability(model : nx.DiGraph, debug = False):
    m = gp.Model("lp")
    p = {s : m.addVar(ub=1.0, name=s, lb = 0) for s in model.nodes}
    
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
            assert len(enabled_actions) <= 1, f'More than one action for non-user state {s} : {enabled_actions}' 
        else:
            assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
        for a in enabled_actions:
            m.addConstr(p[s] <= sum([float(model.edges[e]['prob_weight']) * p[e[1]] for e in model.edges(s) if model.edges[e]['action'] == a]))
        # m.addConstr(p[s] >= sum([float(model.edges[e]['prob_weight']) * p[e[1]] for e in model.edges(s)]))

    # encode reachability constraint
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    
    m.setObjective(p[start_state], sense = GRB.MAXIMIZE)
    m.optimize()
    if debug:
        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")
        print(f"Obj: {m.ObjVal:g}")
    
    return m.ObjVal, construct_optimal_strategy_from_solution(model, p)
    
def z3_feasible(model : nx.DiGraph, target_prob : float, user_strategy : dict, target_d_0 , timeout = 60*60, threads = 1, debug = False):
    # z3.memory_max_size = 8000
    solver = Solver()
    # solver = Optimize()
    p = {s : Real(name=s) for s in model.nodes}
    for v in p:
        solver.add(p[v] <= 1)
        solver.add(0 <= p[v])
    # encode actions
    p_sa = {}
    for s in model.nodes:
        if 'positive' in s:
            solver.add(p[s] == 0)
            continue
        if 'negative' in s:
            solver.add(p[s] == 1)
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        if 'customer' not in s:
            assert len(enabled_actions) <= 1, f'More than one action for non-user state {s} : {enabled_actions}' 
        p_sa[s] = {a : Real(name=s+'_'+a) for a in enabled_actions}
        solver.add(sum(list(p_sa[s].values())) == 1) # scheduler sums up to one
        for a in enabled_actions:
            solver.add(p_sa[s][a] <= 1)
            solver.add(0 <= p_sa[s][a])
            
    # encode model
    for s in p_sa:
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
        solver.add(p[s] == sum([p_sa[s][model.edges[e]['action']] * float(model.edges[e]['prob_weight']) * p[e[1]] for e in model.edges(s)]))

    # encode reachability constraint
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    solver.add(p[start_state] <= target_prob)
    
    def abs(x):
        return If(x >= 0,x,-x)
    
    def add_abs(var, prob, constr):
        solver.add(prob - constr <= var)
        solver.add(constr - prob <= var)

    d_sa = {}
    for s in model.nodes:
        if 'positive' in s or 'negative' in s:
            continue
        if model.edges[list(model.edges(s))[0]]['controllable'] or model.edges[list(model.edges(s))[0]]['action'] == 'env':
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        if 'customer' not in s:
            assert len(enabled_actions) == 1, f'More than one action for non-user state {s}' 
        d_sa[s] = {a : Real(name=f'Abs dist state {s} action {a}') for a in enabled_actions}
        for a in enabled_actions:
            solver.add(d_sa[s][a] <= 1)
            solver.add(0 <= d_sa[s][a])
            add_abs(d_sa[s][a], p_sa[s][a], round(user_strategy[s][a], 2))
            # solver.add(d_sa[s][a] == abs(p_sa[s][a] - user_strategy[s][a]))
        # encode d_inf constraint
    
    # encode sparsity
    decision_changed = {}
    dist_binary = {}
    for s in d_sa:
        dist_binary[s] = Bool(name=f'State {s} was changed')
        decision_changed[s] = Real(name=f'Var dist state {s}')
        solver.add(decision_changed[s] <= 1)
        solver.add(0 <= decision_changed[s])
        solver.add(decision_changed[s] == 0.5 * sum(d_sa[s].values()))
        solver.add(decision_changed[s] <= 10 * dist_binary[s])
    # solver.minimize(sum(dist_binary.values()))
    solver.add(sum(dist_binary.values()) == target_d_0)
        
    if debug or True:
        # set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
        print(solver)
        # print(solver)
       
    r = solver.check() 
    print(r)
    print(solver.statistics())
    
    if r == "sat":
        m = solver.model()
    
def quadratic_program(model : nx.DiGraph, target_prob : float, user_strategy : dict, timeout = 60*60, threads = 1, debug = False):
    gp.setParam("Threads", threads)
    m = gp.Model("qp")
    m.setParam('TimeLimit', timeout)
    m.setParam('SoftMemLimit', 2.5)
    p = {s : m.addVar(ub=1.0, name=s, lb = 0) for s in model.nodes}

    # encode actions
    p_sa = {}
    for s in model.nodes:
        if 'positive' in s:
            m.addConstr(p[s] == 0)
            continue
        if 'negative' in s:
            m.addConstr(p[s] == 1)
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        if 'customer' not in s:
            assert len(enabled_actions) <= 1, f'More than one action for non-user state {s} : {enabled_actions}' 
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
    m.addConstr(p[start_state] <= target_prob)
    
    
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

    
    m.setObjective(d_0 + d_1 + d_inf, sense = GRB.MINIMIZE)
    m.optimize()
    
    if m.status == GRB.INFEASIBLE:
        return Result(m.Runtime, -0.2, target_prob, {}, timeout, 0, m.status)
    
    if m.status == GRB.TIME_LIMIT:
        print("Distances")
        print("d_inf", d_inf.X)
        print("d_1", d_1.X)
        print("d_0", d_0.X)
        return Result(m.Runtime, m.ObjVal, target_prob, {}, timeout, m.MIPGap, m.status)
     
    assert m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL, f'Status is {m.status}'
    print("Distances")
    print("d_inf", d_inf.X)
    print("d_1", d_1.X)
    print("d_0", d_0.X)
    if debug:
        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")
        print(f"Obj: {m.ObjVal:g}")
    
    strategy = construct_strategy_from_solution(model, p_sa, gurobi_access)
    if debug:
        print('Constructed solution')
        print(strategy)
    
    strategy_diff(user_strategy, strategy)
    
    return Result(m.Runtime, m.ObjVal, target_prob, strategy, timeout, m.MIPGap, m.status)

def strategy_slack(strategy):
    total = 0
    for s in strategy:
        total += 1 - sum(strategy[s][a] for a in strategy[s])
    return total

def geometric_program_bnb(model : nx.DiGraph, target_prob : float, user_strategy : dict, changeable_states : list, optimal_strat, timeout = 60*60, debug = False):
    assert 'MOSEK'  in cp.installed_solvers()
    constraints = []
    
    p = {s : cp.Variable(pos=True, name=s) for s in model.nodes}
    
    negative_state = [s for s in model.nodes if "negative" in s]
    assert len(negative_state) == 1
    negative_state = negative_state[0]
    unreaching = [s for s in model.nodes if not nx.has_path(model, s, negative_state)]

    increasing_sa = {} # does not contain pos or negative state, and only controllable states
    for s in optimal_strat:
        increasing_sa[s] = {}
        for a in optimal_strat[s]:
            if user_strategy[s][a] < optimal_strat[s][a] and s in changeable_states:
                increasing_sa[s][a] = True
            else:
                increasing_sa[s][a] = False
    
    # encode actions
    p_sa = {}
    for s in model.nodes:
        if s in unreaching:
            # can't have 0 constraint, have to replace variable with 0
            # m.addConstr(p[s] == 0)
            continue
        if 'negative' in s:
            constraints.append(p[s] == 1)
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        if 'customer' not in s:
            assert len(enabled_actions) <= 1, f'More than one action for non-user state {s} : {enabled_actions}' 
        p_sa[s] = {a : cp.Variable(pos=True, name=s+'_'+a) for a in enabled_actions}
        if len(p_sa[s]) > 1 and s not in changeable_states:
            for a in enabled_actions:
                constraints.append(p_sa[s][a] == user_strategy[s][a])
        else: 
            for a in enabled_actions:
                assert not(s in increasing_sa) or a in increasing_sa[s], f'Action {a} not contained under state {s}'
                if s in increasing_sa and increasing_sa[s][a]:
                    p_sa[s][a] = p_sa[s][a] + user_strategy[s][a]
                    if debug:
                        print("increase", p_sa[s][a])
                constraints.append(p_sa[s][a] <= 1)
            constraints.append(sum(list(p_sa[s].values())) <= 1) # scheduler sums up to one
        # dont allow slack in fixed variables
        if len(p_sa[s]) == 1: # if only one decision, it must receive prob. 1
            assert isinstance(list(p_sa[s].values())[0], type(list(p_sa[s][a].variables())[0])) 
            constraints.append(list(p_sa[s].values())[0] == 1)
            
    # encode model
    for s in p_sa:
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
        assert not all([e[1] in unreaching for e in model.edges(s)]), f'{s} should NOT be contained'
        # insert states leading to 0 as they are replaced by the 0
        s_sum = sum([p_sa[s][model.edges[e]['action']] * float(model.edges[e]['prob_weight']) * p[e[1]] for e in model.edges(s) if e[1] not in unreaching])
        constraints.append(p[s] >= s_sum)
    
    # encode reachability constraint
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    constraints.append(p[start_state] <= target_prob)
    
    d_1 = cp.Variable(pos=True, name='d1')
    d_inf = cp.Variable(pos=True, name='d_inf')

    def get_var(s, a):
        assert s in p_sa, f'{s} not in p_sa'
        assert a in p_sa[s], f'{a} not in p_sa[s] {p_sa[s]}'
        assert len(set(p_sa[s][a].variables())) == 1, f'Found variables {set(p_sa[s][a].variables())}'
        return list(p_sa[s][a].variables())[0]
    
    # strict proximal
    for s in increasing_sa:
        if s in unreaching: # does not have a variable
            continue
        increasing = [get_var(s,a) for a in p_sa[s] if increasing_sa[s][a]]
        if increasing:
            constraints.append(d_inf >= sum(increasing))
    
    # relaxed proximal
    increasing = [get_var(s,a) for s in increasing_sa if s not in unreaching for a in increasing_sa[s] if increasing_sa[s][a]]
    if increasing:
        constraints.append(d_1 >= sum(increasing) / len(user_strategy))
    
    # ensure well-formedness
    for constraint in constraints:
        assert constraint.is_dgp(), constraint
        
    
    problem = cp.Problem(cp.Minimize(sum([1/get_var(s,a) for s in p_sa if s in increasing_sa for a in p_sa[s]]) + d_inf + d_1), constraints)
    # problem = cp.Problem(cp.Minimize(sum([1/get_var(s,a) for s in p_sa for a in p_sa[s]]) + d_inf + d_1), constraints)
    
    if debug:
        print(problem)
        print("Is this problem DGP?", problem.is_dgp())
    assert problem.is_dgp(), "Problem is not DGP"

    precision = 0.0001
    mosek_params={mosek.dparam.intpnt_co_tol_pfeas : precision, mosek.dparam.intpnt_co_tol_pfeas : precision, mosek.dparam.intpnt_co_tol_rel_gap : precision, mosek.dparam.intpnt_co_tol_infeas : precision}
    problem.solve(gp=True, solver="MOSEK", verbose = debug, mosek_params=mosek_params) 
    
    if problem.status != 'optimal':
        print(f'Problem is {problem.status}')
        return Result(problem.solver_stats.solve_time, -0.2, target_prob, {})
    print("sol", problem.value, "in sec.", problem.solver_stats.solve_time)
    
    if debug:
        for s in p_sa:
            print(p[s].value)
            for a in p_sa[s]:
                print(f'state {s} action {a}', p_sa[s][a].value)
            
    print("Distances")
    print("d_inf", d_inf.value)
    print("d_1", d_1.value)
    print("d_0", len(changeable_states))
    
    strategy = construct_strategy_from_solution(model, p_sa, mosek_access, user_strategy)
    
    if debug:
        print('Constructed solution')
        print(strategy)
    
        strategy_diff(user_strategy, strategy)
    
    if strategy_slack(strategy) > STRATEGY_SLACK:
        print('max', max([1 - sum(strategy[s][a] for a in strategy[s]) for s in strategy]))
        print(f'Discarded solution due to slack {strategy_slack(strategy)}')
        assert(False)
        return Result(problem.solver_stats.solve_time, -0.5, target_prob, strategy)
    print("Found solution")
    print()
    return Result(problem.solver_stats.solve_time, d_inf.value + d_1.value + len(changeable_states), target_prob, strategy)

def geometric_program(model : nx.DiGraph, target_prob : float, user_strategy : dict, timeout = 60*60, debug = False):
    assert 'MOSEK'  in cp.installed_solvers()
    
    search_time = 0
    
    changeable_states = [s for s in user_strategy if len(user_strategy[s]) > 1]
    # compute which values to increase
    val, optimal_strat = minimum_reachability(model)
    
    # infeasible
    r = geometric_program_bnb(model, target_prob, user_strategy, changeable_states, optimal_strat, timeout=timeout, debug=debug)
    search_time += r.time
    if r.value < 0:
        return Result(search_time, -0.2, target_prob, {})
    # trivial
    r = geometric_program_bnb(model, target_prob, user_strategy, [], optimal_strat, timeout=timeout, debug=debug)
    search_time += r.time
    if r.value >= 0:
        r.time = search_time
        return r
    
    #lower = 0
    #upper = len(changeable_states)
    #while lower != upper:
    for i in range(len(changeable_states)+1):
        results = {}
        for comb in itertools.combinations(changeable_states, i):
            if search_time > timeout:
                if results:
                    best_solution = min(results.items(), key=lambda x: x[1].value)[1]
                    best_solution.time = search_time
                    return best_solution
                else:
                    return Result(search_time, -0.2, target_prob, {})
            print("Called with changeable states", comb, "from", len(changeable_states), "variables - search time", search_time)
            r = geometric_program_bnb(model, target_prob, user_strategy, comb, optimal_strat, timeout=timeout, debug=debug)
            search_time += r.time
            if r.value > 0:
                results[comb] = r
        if results:
            best_solution = min(results.items(), key=lambda x: x[1].value)[1]
            best_solution.time = search_time
            return best_solution
    return Result(search_time, -0.2, target_prob, {})
    
        
    constraints = []
    
    p = {s : cp.Variable(pos=True, name=s) for s in model.nodes}
    
    # compute which values to increase
    val, optimal_strat = minimum_reachability(model)
    
    negative_state = [s for s in model.nodes if "negative" in s]
    assert len(negative_state) == 1
    negative_state = negative_state[0]
    unreaching = [s for s in model.nodes if not nx.has_path(model, s, negative_state)]

    increasing_sa = {} # does not contain pos or negative state, and only controllable states
    for s in optimal_strat:
        increasing_sa[s] = {}
        for a in optimal_strat[s]:
            if user_strategy[s][a] < optimal_strat[s][a]:
                increasing_sa[s][a] = True
            else:
                increasing_sa[s][a] = False
    
    # encode actions
    p_sa = {}
    for s in model.nodes:
        if s in unreaching:
            # can't have 0 constraint, have to replace variable with 0
            # m.addConstr(p[s] == 0)
            continue
        if 'negative' in s:
            constraints.append(p[s] == 1)
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        if 'customer' not in s:
            assert len(enabled_actions) <= 1, f'More than one action for non-user state {s} : {enabled_actions}' 
        p_sa[s] = {a : cp.Variable(pos=True, name=s+'_'+a) for a in enabled_actions}
        for a in enabled_actions:
            assert not(s in increasing_sa) or a in increasing_sa[s], f'Action {a} not contained under state {s}'
            if s in increasing_sa and increasing_sa[s][a]:
                p_sa[s][a] = p_sa[s][a] + user_strategy[s][a]
                if debug:
                    print("increase", p_sa[s][a])
            constraints.append(p_sa[s][a] <= 1)
        constraints.append(sum(list(p_sa[s].values())) <= 1) # scheduler sums up to one
        # dont allow slack in fixed variables
        if len(p_sa[s]) == 1: # if only one decision, it must receive prob. 1
            assert isinstance(list(p_sa[s].values())[0], type(list(p_sa[s][a].variables())[0])) 
            constraints.append(list(p_sa[s].values())[0] == 1)
            
    # encode model
    for s in p_sa:
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
        assert not all([e[1] in unreaching for e in model.edges(s)]), f'{s} should NOT be contained'
        # insert states leading to 0 as they are replaced by the 0
        s_sum = sum([p_sa[s][model.edges[e]['action']] * float(model.edges[e]['prob_weight']) * p[e[1]] for e in model.edges(s) if e[1] not in unreaching])
        constraints.append(p[s] >= s_sum)
    
    # encode reachability constraint
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    constraints.append(p[start_state] <= target_prob)
    
    d_0 = cp.Variable(pos=True, name='d0')
    d_1 = cp.Variable(pos=True, name='d1')
    d_inf = cp.Variable(pos=True, name='d_inf')

    def get_var(s, a):
        assert s in p_sa, f'{s} not in p_sa'
        assert a in p_sa[s], f'{a} not in p_sa[s] {p_sa[s]}'
        assert len(set(p_sa[s][a].variables())) == 1, f'Found variables {set(p_sa[s][a].variables())}'
        return list(p_sa[s][a].variables())[0]
    
    # strict proximal
    for s in increasing_sa:
        if s in unreaching: # does not have a variable
            continue
        increasing = [get_var(s,a) for a in p_sa[s] if increasing_sa[s][a]]
        if increasing:
            constraints.append(d_inf >= sum(increasing))
    
    # relaxed proximal
    increasing = [get_var(s,a) for s in increasing_sa if s not in unreaching for a in increasing_sa[s] if increasing_sa[s][a]]
    print("increasing", increasing)
    constraints.append(d_1 >= sum(increasing) / len(user_strategy))
    
    # ensure well-formedness
    for constraint in constraints:
        assert constraint.is_dgp(), constraint
        
    print(sum([1/get_var(s,a) for s in p_sa if s in increasing_sa for a in p_sa[s]]))
    problem = cp.Problem(cp.Minimize(sum([1/get_var(s,a) for s in p_sa if s in increasing_sa for a in p_sa[s]]) + d_inf + d_1), constraints)
    # problem = cp.Problem(cp.Minimize(sum([1/get_var(s,a) for s in p_sa for a in p_sa[s]]) + d_inf + d_1), constraints)
    
    if debug:
        print(problem)
        print("Is this problem DGP?", problem.is_dgp())
    assert problem.is_dgp(), "Problem is not DGP"

    problem.solve(gp=True, solver="MOSEK", verbose = True)
    
    if problem.status != 'optimal':
        return Result(problem.solver_stats.solve_time, -0.2, target_prob, {})
    print("sol", problem.value, "in sec.", problem.solver_stats.solve_time)
    
    if debug:
        for s in p_sa:
            print(p[s].value)
            for a in p_sa[s]:
                print(f'state {s} action {a}', p_sa[s][a].value)
            
    print("Distances")
    print("d_inf", d_inf.value)
    print("d_1", d_1.value)
    
    strategy = construct_strategy_from_solution(model, p_sa, mosek_access, user_strategy)
    
    if debug:
        print('Constructed solution')
        print(strategy)
    
    strategy_diff(user_strategy, strategy)
    
    if strategy_slack(strategy) > STRATEGY_SLACK:
        print(f'Discarded solution due to slack {strategy_slack(strategy)}')
        return Result(problem.solver_stats.solve_time, -0.5, target_prob, strategy)
    
    return Result(problem.solver_stats.solve_time, d_inf.value + d_1.value, target_prob, strategy)     
        
def construct_user_strategy(model : nx.DiGraph):
    user_strategy = {}
    
    for s in model.nodes:
        if "positive" in s or "negative" in s:
            continue
        if not model.edges[list(model.edges(s))[0]]['controllable'] and model.edges[list(model.edges(s))[0]]['action'] != 'env':
            enabled_actions = list(set([model.edges[e]['action'] for e in model.edges(s)]))
            distr = random.choices(range(1, 1000), k=len(enabled_actions))
            user_strategy[s] = {enabled_actions[i] : round(distr[i] / sum(distr), 4) for i in range(len(enabled_actions))}
    
    return user_strategy

def strategy_diff(strat1 : dict, strat2 : dict):
    assert strat1.keys() == strat2.keys()
    for s in strat1:
        assert s in strat2
        assert strat1[s].keys() == strat2[s].keys()
        for a in strat1[s]:
            if round(strat1[s][a], 2) != round(strat2[s][a], 2):
                print(f'In state {s} action {a} differs, {strat1[s][a]} != {strat2[s][a]}')

def plot_results(geom, qp, optimal, experiments):
    assert len(geom) == len(qp), f'len(geom){len(geom)} != len(qp){len(qp)}'
    
    fig = plt.figure()
    
    for i in range(len(experiments)):
        ax = plt.subplot(len(experiments), 2, 2*i+1)
        ax.set_title(experiments[i] + "-value")
        for j in range(len(geom[i])):
            #ax.plot([r.target_prob for r in geom[i][j]], [r.value for r in geom[i][j]], c = "blue", label="GP" if j == 0 else '', linewidth = 1, marker='o')
            ax.plot([r.target_prob for r in qp[i][j]], [r.value for r in qp[i][j]], c = "orange", label="QP" if j == 0 else '', linewidth = 1, marker='*')
            ax.axvline([optimal[i][j][0]], c = 'violet', linestyle='--')
        ax.legend()
        
    for i in range(len(experiments)):
        ax = plt.subplot(len(experiments), 2, 2*i+2)
        ax.set_title(experiments[i] + "-time")
        for j in range(len(geom[i])):
            #ax.plot([r.target_prob for r in geom[i][j]], [r.time for r in geom[i][j]], c = "blue", label="GP" if j == 0 else '', linewidth = 1, marker='o')
            ax.plot([r.target_prob for r in qp[i][j]], [r.time for r in qp[i][j]], c = "orange", label="QP" if j == 0 else '', linewidth = 1, marker='*')
        ax.axvline([optimal[i][j][0]], c = 'violet', linestyle='--')
        ax.legend()
        
    plt.savefig('out/plot.png', dpi = 500)

def plot_changes(model : nx.DiGraph, name : str, user_strategy, counterfactual_strategy, layout = "sfdp"):
    #def draw_dfg(g, name, names={}, layout = "sfdp", color_map = [], add_greps_cluster=True):
    """
    Helper function to draw Networkx graphs.
    """
    scaling = 10
    # build graph with variable thickness
    #scaling = 1/np.mean(list(nx.get_edge_attributes(g,'edge_weight').values()))

    A = to_agraph(model)

    edge_weights = nx.get_edge_attributes(model,'edge_weight')
    for e in edge_weights:
        e = A.get_edge(e[0], e[1])
        # e.attr["penwidth"] = edge_weights[e]*scaling
        # e.attr["fontsize"] = "20"
    for e in model.edges:
        edge = A.get_edge(e[0], e[1])
        if 'controllable' in model[e[0]][e[1]]:
            if not model[e[0]][e[1]]['controllable']:
                edge.attr["style"] = "dotted"
                #edge.attr["label"] =  str(g[e[0]][e[1]]["prob_weight"])
        #A.add_edge(e[0], e[1], penwidth = edge_weights[e]*scaling)

    for n in A.nodes():
        n.attr['label'] = n.split(':')[1]
        # n.attr['fontsize'] = 120
        # n.attr['penwidth'] = 30
        # n.attr['height'] = 3
        # n.attr['width'] = 3

    for e in A.edges():
        # e.attr['penwidth'] = 20
        # e.attr["fontsize"] = 120
        e.attr["label"] = str(round(model[e[0]][e[1]]["prob_weight"],2))
        e.attr["color"] = "black"
        
    for s in user_strategy:
        for e in model.edges(s):
            e_a = A.get_edge(e[0], e[1])
            if any([round(user_strategy[s][a], 2) != round(counterfactual_strategy[s][a], 2) for a in user_strategy[s]]):
                e_a.attr["color"] ="red"
                n = A.get_node(s)
                n.attr['color'] = 'red'

      
    A.write(f'out/{name}.dot')
    A.layout(layout)
    A.draw(f'out/{name}.png')
    print("Plotted", name)

def generate_models(experiments, cores = 1, model_iterations = 1):
    parser_list = []
    for name in experiments:
        if name == 'greps':
            print("######### GREPS ##########")
            from LogParser import GrepsParser
            parser = GrepsParser('data/data.csv', 'data/activities_greps.xml')
        elif name == 'bpic12':
            print("######### BPIC'12 ##########")
            from LogParser import BPIC12Parser
            parser = BPIC12Parser('data/BPI_Challenge_2012.xes', 'data/activities_2012.xml')
        elif name == 'bpic17-before':
            print("######### BPIC'17-Before ##########")
            from LogParser import BPIC17BeforeParser
            parser = BPIC17BeforeParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
        elif name == 'bpic17-after':
            print("######### BPIC'17-After ##########")
            from LogParser import BPIC17AfterParser
            parser = BPIC17AfterParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
        elif name == 'bpic17-both':
            print("######### BPIC'17-Both ##########")
            from LogParser import BPIC17BothParser
            parser = BPIC17BothParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
        elif 'spotify' in name:
            print("######### Spotify ##########")
            from LogParser import SpotifyParser
            parser = SpotifyParser('data/spotify/', 'data/activities_spotify.xml', int(name.split('spotify')[1]))
        else:
            continue
        
        if type(parser) != list:
            parser_list.extend([(parser, i, name) for i in range(model_iterations)])
        else:
            parser_list.extend([(p, i, name) for p in parser for i in range(model_iterations)])
                    
    #[build_and_write_model(p) for p in parser]
    with multiprocessing.Pool(processes=cores) as pool:
        pool.map(build_and_write_model, parser_list)
            
def build_and_write_model(input):
    parser = input[0]
    i = input[1]
    name = input[2]
    model = parser.build_benchmark()
    print(type(model))
    with open(f'out/models/model_{name}_model-it_{i}.pickle', 'wb+') as handle:
        pickle.dump(model, handle)
                    
def generate_user_strategies(experiments, model_iterations, iterations):
    for name in experiments:
        for i in range(model_iterations):
            with open(f'out/models/model_{name}_model-it_{i}.pickle', 'rb') as handle:
                model = pickle.load(handle)
            for j in range(iterations):
                user_strategy = construct_user_strategy(model) # NOTE strategies are not guaranteed to be equivalent as the set construction can change
                with open(f'out/user_strategies/model_{name}_model-it_{i}_it_{j}.pickle', 'wb+') as handle:
                        pickle.dump(user_strategy, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_experiment(parser : LogParser.LogParser, timeout, name = "model", steps = 1, iterations = 1):
    model = parser.build_benchmark()
    if REGENERATE or True:
        with open(f'out/model_{name}.pickle', 'wb+') as handle:
            pickle.dump(model, handle)
        user_strategy = construct_user_strategy(model) # NOTE strategies are not guaranteed to be equivalent as the set construction can change
        with open(f'out/user_strategy_{name}.pickle', 'wb+') as handle:
            pickle.dump(user_strategy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'out/model_{name}.pickle', 'rb') as handle:
        model = pickle.load(handle)
    with open(f'out/user_strategy_{name}.pickle', 'rb') as handle:
        user_strategy = pickle.load(handle)
    r_geom = []
    r_qp = []
    for it in range(iterations):
        r_geom_it = []
        r_qp_it = []
        user_strategy = construct_user_strategy(model)
        o, strat = minimum_reachability(model)
        print("optimal", o)
        for i in range(0, steps+1):
            # p = 1 : trivial, p = 0 : impossible
            # p = 1 - i * o / steps
            p = 1/(steps)*i
            if p == 0:
                p = 0.0001
            print(f'Call with reachability probability {p}')
            r_geom_it.append(geometric_program(model,p, user_strategy, timeout=timeout))
            r_qp_it.append(quadratic_program(model, p, user_strategy, timeout=timeout))
            #plot_changes(model, 'diff', r_geom[-1].strategy, user_strategy, layout='dot')
        r_geom.append(r_geom_it)
        r_qp.append(r_qp_it)
    
    return r_geom, r_qp, o

def search_bounds(model, user_strategy, debug = False):
    o, optimal_strat = minimum_reachability(model)
    p = 1
    while(p > o + 0.00001):
        print("####### call with p = ", p)
        r = quadratic_program(model, p, user_strategy, timeout = 5, debug=debug)
        if r.value == 0:
            # reduce p
            p = (o + p)/2
        else:
            return (max(0.001, o-0.1),(p+0.1))
    return (0.001,1)

def round_probabilities(model):
    for s in model.nodes:
        total_sum = sum([round(float(model.edges[e]['prob_weight']), 2) for e in model.edges(s)])
        for e in model.edges(s):
            model.edges[e]['prob_weight'] = round(float(model.edges[e]['prob_weight']), 2) / total_sum
    return model

def run_experiment(param):
    path = param[0]
    p = param[1]
    if p == 0:
        p = 0.0001
    timeout = param[2]
    
    model_path = str(path).split('_it_')[0].replace('user_strategies', 'models')
    with open(f'{model_path}.pickle', 'rb') as handle:
        model = pickle.load(handle)
    with open(path, 'rb') as handle:
        user_strategy = pickle.load(handle)
        
    o, strat = minimum_reachability(model)
    
    print(f'Call {model_path} with reachability probability {p} on strategy {path}')
    r_geom = Result(0, 0, 0, {}, 0, 0) # geometric_program(model,p, user_strategy, timeout=timeout, debug=True)
    r_qp = quadratic_program(model, p, user_strategy, timeout=timeout, debug=False)
    o, strat = minimum_reachability(model)

    return (path, p, r_geom, r_qp, o)

    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(
                    prog = 'benchmarks',
                    description = "File to trigger benchmarks for CE generation in MDP's")
    parser.add_argument('-t', '--timeout', help = "Timeout for program solution", type=int, default = 60*60) 
    parser.add_argument('-s', '--steps', help = "Number of steps for each model", type=int, default = 1)
    parser.add_argument('-i', '--iterations', help = "Iterations for each step", type=int, default = 1)
    parser.add_argument('-c', '--cores', help = "Cores to use to parallelize experiments", type=int, default = 1)
    parser.add_argument('-slack', '--strategy_slack', help = "Allowed deviation until geometric programming strategy is discarded", type=float, default = 0.1)
    parser.add_argument('-e', '--experiments', help = "Start profile to filter on", nargs='+', type=str, default = ['greps', 'bpic12', 'bpic17-before', 'bpic17-after', 'bpic17-both', 'spotify'])
    parser.add_argument('-rm', '--rebuild_models', help = "Rebuild models, implies rebuilding models", action = 'store_true')
    parser.add_argument('-rs', '--rebuild_strategies', help = "Rebuild strategies", action = 'store_true')
    parser.add_argument('-mi', '--model_iterations', help = "Number of models to generate for each setting", type=int, default = 10)
    parser.add_argument('-as', '--all_spotify', help = "All spotify models in steps of 100 are generated", action = 'store_true')
    args = parser.parse_args()
    
    STRATEGY_SLACK = args.strategy_slack
    
    geom_results = []
    qp_results = []
    optimal_reachability = []
    
    if args.all_spotify:
        args.experiments.remove('spotify')
        args.experiments.extend([f'spotify{i*1000}' for i in range(1,11)])
    
    if args.rebuild_models:
        filename = "out/models/test.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        generate_models(args.experiments, args.cores, args.model_iterations)
    if args.rebuild_models or args.rebuild_strategies:
        filename = "out/user_strategies/test.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        generate_user_strategies(args.experiments, args.model_iterations, args.iterations)
    
    
    # for name in args.experiments:
    benchmark_strategies = []
    for name in args.experiments:
        for i in range(args.iterations):
            for j in range(args.model_iterations):
                assert list(Path(f'out/user_strategies/').glob(f'*{name}_model-it_{j}*_it_{i}*.pickle'))
                benchmark_strategies.extend(list(Path(f'out/user_strategies/').glob(f'*{name}_model-it_{j}*_it_{i}*.pickle')))
    
    experiments = []
    for e in benchmark_strategies:
        path = str(e).split('_it_')[0].replace('user_strategies', 'models')
        name = str(e).split('model_')[1].split('_')[0]
        with open(f'{path}.pickle', 'rb') as handle: #out/models/model_{name}
            model = round_probabilities(pickle.load(handle))
        with open(e, 'rb') as handle:
            user_strategy = pickle.load(handle)   
        bounds = (0,1)#search_bounds(model, user_strategy)
        print(bounds)
        experiments.extend([(e, round(bounds[0] + (bounds[1] - bounds[0]) * 1/(args.steps) * s, 4), args.timeout) for s in range(args.steps+1)])
        experiments.append((e, 1, args.timeout))
    # experiments = [(p, 1/(args.steps)*s, args.timeout) for p in benchmark_strategies for s in range(args.steps+1)]

    # manual tests
    path = 'out/models/model_spotify1000_model-it_0.pickle'
    name = str(path).split('model_')[1].split('_')[0]
    with open('out/models/model_spotify3000_model-it_0.pickle', 'rb') as handle: #open(f'out/models/model_{name}.pickle', 'rb') as handle:
        model = pickle.load(handle)
    with open('out/user_strategies/model_spotify3000_model-it_0_it_0.pickle', 'rb') as handle:
        user_strategy = pickle.load(handle)
        # user_strategy = pickle.load(handle)   
    # print("search_bounds", search_bounds(model, user_strategy))
    
    o, strat = minimum_reachability(model)
    print("optimal", o)
    r_qp = z3_feasible(model, 0.35, user_strategy, 1, timeout=args.timeout, debug=False)
    r_qp = quadratic_program(model, 0.35, user_strategy, timeout=args.timeout, debug=False)
    print(r_qp.df())
    # run_experiment((path, 0.35, args.timeout))
    assert(False)
    
    df_results = pd.DataFrame()
    stored_results = []
    with multiprocessing.Pool(processes=args.cores) as pool:
        result = pool.imap_unordered(run_experiment, experiments)
        for r in result:
            stored_results.append(r)
            new_df = r[3].df()
            new_df['path'] = [r[0]]
            df_results = pd.concat([df_results, new_df])
            df_results.to_csv("out/results.csv")
    # result = [run_experiment(e) for e in experiments]
    result = stored_results
    
    assert(False)
    
    r_geom = []
    r_qp = []
    r_o = []
    for i in range(len(args.experiments)):
        name = args.experiments[i]
        r_geom_name = []
        r_qp_name = []
        r_o_name = []
        for j in range(args.iterations):
            r_geom_inner = []
            r_qp_inner = []
            r_o_inner = []
            for k in range(args.steps+2):
                r_geom_inner.append(result[i*(args.iterations * (args.steps + 2)) + j * (args.steps + 2) + k][2])
                r_qp_inner.append(result[i*(args.iterations * (args.steps + 2)) + j * (args.steps + 2) + k][3])
                r_o_inner.append(result[i*(args.iterations * (args.steps + 2)) + j * (args.steps + 2) + k][4])
            r_geom_name.append(r_geom_inner)
            r_qp_name.append(r_qp_inner)
            r_o_name.append(r_o_inner)
        r_geom.append(r_geom_name)
        r_qp.append(r_qp_name)
        r_o.append(r_o_name)
    plot_results(r_geom, r_qp, r_o, args.experiments)     
    assert(False)
    
    for name in args.experiments:
        if name == 'greps':
            print("######### GREPS ##########")
            from LogParser import GrepsParser
            parser = GrepsParser('data/data.csv', 'data/activities_greps.xml')
        elif name == 'bpic12':
            print("######### BPIC'12 ##########")
            from LogParser import BPIC12Parser
            parser = BPIC12Parser('data/BPI_Challenge_2012.xes', 'data/activities_2012.xml')
        elif name == 'bpic17-before':
            print("######### BPIC'17-Before ##########")
            from LogParser import BPIC17BeforeParser
            parser = BPIC17BeforeParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
        elif name == 'bpic17-after':
            print("######### BPIC'17-After ##########")
            from LogParser import BPIC17AfterParser
            parser = BPIC17AfterParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
        elif name == 'bpic17-both':
            print("######### BPIC'17-Both ##########")
            from LogParser import BPIC17BothParser
            parser = BPIC17BothParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml')
        elif name == 'spotify':
            print("######### Spotify ##########")
            from LogParser import SpotifyParser
            parser = SpotifyParser('data/spotify/', 'data/activities_spotify.xml')
        else:
            continue
        r_geom, r_qp, o = run_experiment(parser, timeout=args.timeout, steps=args.steps, iterations=args.iterations)
        geom_results.append(r_geom)
        qp_results.append(r_qp)
        optimal_reachability.append(o)

    # # greps example
    # if 'greps' in args.experiments:
    #     pass
        
    # # BPIC'12 example
    # if 'bpic12' in args.experiments:
    #     print("######### BPIC'12 ##########")
    #     from LogParser import BPIC12Parser
    #     r_geom, r_qp = run_experiment(BPIC12Parser('data/BPI_Challenge_2012.xes', 'data/activities_2012.xml'), 0.7, timeout=args.timeout, steps=args.steps)
    #     geom_results.append(r_geom)
    #     qp_results.append(r_qp)
        
    # # BPIC'17 example
    # if 'bpic17-before' in args.experiments:
    #     print("######### BPIC'17-Before ##########")
    #     from LogParser import BPIC17BeforeParser
    #     r_geom, r_qp = run_experiment(BPIC17BeforeParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml'), 0.7, timeout=args.timeout, steps=args.steps)
    #     geom_results.append(r_geom)
    #     qp_results.append(r_qp)
        
    # if 'bpic17-after' in args.experiments:
    #     print("######### BPIC'17-After ##########")
    #     from LogParser import BPIC17AfterParser
    #     r_geom, r_qp = run_experiment(BPIC17AfterParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml'), 0.7, timeout=args.timeout, steps=args.steps)
    #     geom_results.append(r_geom)
    #     qp_results.append(r_qp)
        
    # if 'bpic17-both' in args.experiments:
    #     print("######### BPIC'17-Both ##########")
    #     from LogParser import BPIC17BothParser
    #     r_geom, r_qp = run_experiment(BPIC17BothParser('data/BPI Challenge 2017.xes', 'data/activities_2017.xml'), 0.7, timeout=args.timeout, steps=args.steps)
    #     geom_results.append(r_geom)
    #     qp_results.append(r_qp)
    
    # # Spotify
    # if 'spotify' in args.experiments:
    #     print("######### Spotify ##########")
    #     from LogParser import SpotifyParser
    #     r_geom, r_qp = run_experiment(SpotifyParser('data/spotify/', 'data/activities_spotify.xml'), 0.35, timeout=args.timeout, steps=args.steps)
    #     geom_results.append(r_geom)
    #     qp_results.append(r_qp)

    plot_results(geom_results, qp_results, optimal_reachability, args.experiments)

# TODO test for spotify
# - how start profile?
# - when end? -> if early than 20 steps
# - switch company and customer
# TODO geom problem d_0 constraints: iterative solution? - even necessary? min sum of changes
# TODO clear file + git commit
# Improving:
# - Repairing strategies
# - Iterative approach 
# TODO write comments