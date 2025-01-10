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
            add_abs(d_sa[s][a], p_sa[s][a], user_strategy[s][a])
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
    
def evaluate_strategy(model : nx.DiGraph, user_strategy : dict, timeout = 60*60, threads = 1, debug = False):
    gp.setParam("Threads", threads)
    m = gp.Model("qp")
    m.setParam('TimeLimit', timeout)
    m.setParam('SoftMemLimit', 2.5)
    p = {s : m.addVar(ub=1.0, name=s, lb = 0) for s in model.nodes}
    
    # encode model
    for s in p:
        if 'positive' in s:
            m.addConstr(p[s] == 0)
            continue
        if 'negative' in s:
            m.addConstr(p[s] == 1)
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
        if 'customer' not in s:
            assert len(enabled_actions) == 1
            m.addConstr(p[s] == sum([float(model.edges[e]['prob_weight']) * p[e[1]] for e in model.edges(s)]))
        else:
            m.addConstr(p[s] == sum([user_strategy[s][model.edges[e]['action']] * float(model.edges[e]['prob_weight']) * p[e[1]] for e in model.edges(s)]))
    
    # encode reachability constraint
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    
    m.setObjective(p[start_state], sense = GRB.MINIMIZE)
    m.optimize()
    
    return m.ObjVal

def expected_visiting_time(model : nx.DiGraph, user_strategy : dict, timeout = 60*60, threads = 5, debug = False):
    env = gp.Env()
    m = gp.Model("qp", env=env)
    m.setParam('TimeLimit', timeout)
    m.setParam('SoftMemLimit', 2.5)
    m.setParam('Threads', threads)
    
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    
    transient_states = list(model.nodes)
    
    induced_model = induced_mc(model, user_strategy)
    
    visits_given = {s : m.addVar(name=f'visiting(given) {s}', lb = 0) for s in model.nodes}
    for s in transient_states:
        # change to quadratic_program: has to consider ingoing edges, not outgoing edges
        if debug:
            print("s", s)
        for e in induced_model.in_edges(s):
            assert s == e[1]
            if debug:
                print(f'{float(induced_model.edges[e]["prob_weight"])} * {e[0]} {visits_given[e[0]]} + {(1 if s == start_state else 0)}')
        m.addConstr(visits_given[s] == sum([float(induced_model.edges[e]['prob_weight']) * visits_given[e[0]] for e in induced_model.in_edges(s)]) + (1 if s == start_state else 0))

    m.setObjective(0, sense = GRB.MINIMIZE)
    m.optimize()
    
    expected_visits = {s : visits_given[s].X for s in transient_states }
    
    m.dispose()
    
    return expected_visits

def induced_mc(model : nx.DiGraph, user_strategy : dict):
    model = copy.deepcopy(model)
    for e in model.edges:
        # For company and environment actions, there is only one action
        if model.edges[e]['controllable'] or model.edges[e]['action'] == 'env':
            enabled_actions = set([model.edges[e]['action'] for e in model.edges(e[0])])
            assert len(enabled_actions) == 1
            continue
        else:
            # normalize transition probabilities with strategy
            model.edges[e]['prob_weight'] = model.edges[e]['prob_weight'] * user_strategy[e[0]][model.edges[e]['action']]
            
    for s in model.nodes:
        if 'positive' in s or 'negative' in s:
            continue
        enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
        assert sum([model.edges[e]['prob_weight'] for e in model.edges(s)]), f'Sum of probabilities at {s} does not sum up to 1'
        
    return model 


def diversity_program(model : nx.DiGraph, target_prob : float, user_strategy : dict, solution : Result, timeout = 60*60, threads = 1, debug = False):
    # bound number of steps via visiting times
    # compare expected visiting times
    # what would be good norm for visiting times?
    # TODO which norms have others used?
    # TODO how multiple results? - combine for n generations?
    gp.setParam("Threads", threads)
    m = gp.Model("qp")
    m.setParam('TimeLimit', timeout)
    m.setParam('SoftMemLimit', 2.5)
    
    # encode reachability constraint
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    
    ##### Previous program QP
    
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
    
    ##############
    
    # TODO solve for transient states
    transient_states = list(model.nodes)
    
    induced_model = induced_mc(model, solution.strategy)
    
    visits_given = {s : m.addVar(name=f'visiting(given) {s}', lb = 0) for s in model.nodes}
    for s in transient_states:
        # change to quadratic_program: has to consider ingoing edges, not outgoing edges
        print("s", s)
        for e in induced_model.in_edges(s):
            assert s == e[1]
            print(f'{float(induced_model.edges[e]["prob_weight"])} * {e[0]} {visits_given[e[0]]} + {(1 if s == start_state else 0)}')
        m.addConstr(visits_given[s] == sum([float(induced_model.edges[e]['prob_weight']) * visits_given[e[0]] for e in induced_model.in_edges(s)]) + (1 if s == start_state else 0))
    
    
    # encode second visiting constraints
    visits = {s : m.addVar(name=f'visiting {s}', lb = 0) for s in transient_states}
    
    for s in transient_states:
        # change to quadratic_program: has to consider ingoing edges, not outgoing edges
        print("s", s)
        for e in model.in_edges(s):
            assert s == e[1]
            print(e, model.edges[e])
            print(f'{float(model.edges[e]["prob_weight"])} * {e[0]} {visits[e[0]]} + {(1 if s == start_state else 0)}')
        ingoing_sum = 0
        for e in model.in_edges(s):
            if e[0] in p_sa:
                ingoing_sum += p_sa[e[0]][model.edges[e]['action']] * float(model.edges[e]['prob_weight']) * visits[e[0]]
            else:
                enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
                if 'positive' in s or 'negative' in s:
                    assert len(enabled_actions) == 0
                else:
                    assert len(enabled_actions) == 1
                ingoing_sum += 1 * float(model.edges[e]['prob_weight']) * visits[e[0]]    
        m.addConstr(visits[s] == ingoing_sum + (1 if s == start_state else 0))
        
    time_to_absorption_var = m.addVar(name='time_to_absorption_var', lb = 0.1)
    m.addConstr(time_to_absorption_var == sum([visits[s] for s in transient_states]))
    
    time_to_absorption_fixed = m.addVar(name='time_to_absorption_fixed', lb = 0.1)
    m.addConstr(time_to_absorption_fixed == sum([visits_given[s] for s in transient_states]))
    
    weighted_visits_given = {s : m.addVar(name=f'weighted visiting(given) {s}', lb = 0) for s in transient_states}
    for s in weighted_visits_given:
        m.addConstr(weighted_visits_given[s] * time_to_absorption_fixed == visits_given[s])
        
    weighted_visits = {s : m.addVar(name=f'weighted visiting {s}', lb = 0) for s in transient_states}
    for s in weighted_visits:
        m.addConstr(weighted_visits[s] * time_to_absorption_var == visits[s])
        
    # # d_0 constraints
    # d_0_visits = m.addVar(name='d_0_visits', lb = 0)
    # # encode binary changes
    # decision_changed_visits = {}
    # dist_binary_visits = {}
    # for s in transient_states:
    #     dist_binary_visits[s] = m.addVar(ub=1.0, name=f'State {s} was changed', lb = 0, vtype=gp.GRB.BINARY)
    #     decision_changed_visits[s] = m.addVar(ub=1.0, name=f'Var dist state {s}', lb = 0) 
    #     m.addConstr(decision_changed_visits[s] >= weighted_visits_given[s] - weighted_visits[s])
    #     m.addConstr(decision_changed_visits[s] >= weighted_visits[s] - weighted_visits_given[s])
    #     m.addConstr(decision_changed_visits[s] <= 10 * dist_binary_visits[s])
    # m.addConstr(sum(dist_binary_visits.values()) == d_0_visits)
    
    d_1_visits = m.addVar(name='d_1_visits')
    decision_changed_visits = {}
    decision_changed_visits_abs = {}
    for s in transient_states:
        decision_changed_visits[s] = m.addVar(name=f'Var dist state {s}', lb = -float('inf'))
        decision_changed_visits_abs[s] = m.addVar(name=f'Var dist state abs {s}', lb = 0)
        m.addConstr(decision_changed_visits[s] == visits_given[s] - visits[s])
        m.addConstr(decision_changed_visits_abs[s] == gp.abs_(decision_changed_visits[s]))
    m.addConstr(sum(decision_changed_visits_abs.values()) == d_1_visits)
    
    # # covariance constraints
    # # WARNING problem: covariance is always small, also if same strategy as distribution of visits is skewed
    # # corr = m.addVar(name='correlation', lb = -1, ub = 1)
    # mean_x = m.addVar(name='mean_x')
    # mean_y = m.addVar(name='mean_y')
    # m.addConstr(mean_x == sum(weighted_visits[s] for s in transient_states) / len(transient_states))
    # m.addConstr(mean_y == sum(weighted_visits_given[s] for s in transient_states) / len(transient_states))
    
    # cov = m.addVar(name='covar')
    # m.addConstr(cov == sum([(weighted_visits[s] - mean_x) * (weighted_visits_given[s] - mean_y) for s in transient_states]) / len(transient_states))
    # # m.addConstr(cov == (sum([(visits[s])  * (visits_given[s]) for s in model.nodes])  / (len(model.nodes) * len(model.nodes))) - 
    #             # (sum([visits[s] for s in model.nodes]) / len(model.nodes)) * (sum([visits_given[s] for s in model.nodes]) / len(model.nodes)) )
                
    # cov_abs = m.addVar(name='abs(covar)', lb = 0)
    # m.addConstr(-cov_abs <= cov)
    # m.addConstr(cov <= cov_abs)
    
    m.setObjective(d_0 + d_1 + d_inf - d_1_visits, sense = GRB.MINIMIZE)
    m.optimize()

    strategy = construct_strategy_from_solution(model, p_sa, gurobi_access)
    if debug:
        print('Constructed solution')
        print(strategy)
    
    if True:
        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")
    print(f"Obj: {m.ObjVal:g}")
    
    strategy_diff(user_strategy, strategy)
    
    assert False
    
# TODO write functions to add individual constraints, taking Gurobi model
def diversity_program_strategy(model : nx.DiGraph, target_prob : float, user_strategy : dict, solutions : list, timeout = 60*60, threads = 1, debug = False):
    # bound number of steps via visiting times
    # compare expected visiting times
    # what would be good norm for visiting times?
    # TODO which norms have others used?
    # TODO how multiple results? - combine for n generations?
    # gp.setParam("Threads", threads)
    env = gp.Env()
    m = gp.Model("qp", env=env)
    m.setParam('TimeLimit', timeout)
    m.setParam('SoftMemLimit', 2.5)
    m.setParam('Threads', threads)
    
    # encode reachability constraint
    start_state = [s for s in model.nodes if 'q0: start' in s]
    assert len(start_state) == 1, start_state
    start_state = start_state[0]
    
    ##### Previous program QP
    
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
    
    ##############
        
    list_of_strategies = [s.strategy for s in solutions]
    list_of_strategies.insert(0, p_sa)
    names_list_of_strategies = [f's{i}' for i in range(len(list_of_strategies)-1)]
    names_list_of_strategies.insert(0, "p_sa")
    
    def det(A):
        v = m.addVar(lb=-float("inf"))
        if A.shape == (2,2):
            m.addConstr(v == A[0,0]*A[1,1]-A[1,0]*A[0,1])
            return v
        expr = gp.QuadExpr()
        cofactor = 1
        for i in range(A.shape[1]):
            cols = [c for c in range(A.shape[1]) if c != i]
            expr += cofactor*A[0,i]*det(A[1:][:,cols])
            cofactor = -cofactor
        m.addConstr(v == expr)
        return v

    # def det(visits):
    #     print(visits[('p_sa', 's0')])
    #     return visits[('p_sa', 's0')]
    
    d_sa_strat = {}
    d_sa_strat_abs = {}
    d_1_visits = m.addMVar((len(list_of_strategies),len(list_of_strategies)), lb=-float("inf"))
    for i in range(len(list_of_strategies)):
        s1 = list_of_strategies[i]
        n1 = names_list_of_strategies[i]
        for j in range(len(list_of_strategies)):
            s2 = list_of_strategies[j]
            n2 = names_list_of_strategies[j]
            d_sa_strat[(n1,n2)] = {}
            d_sa_strat_abs[(n1,n2)] = {}
            for s in model.nodes:
                if 'positive' in s or 'negative' in s:
                    continue
                if model.edges[list(model.edges(s))[0]]['controllable'] or model.edges[list(model.edges(s))[0]]['action'] == 'env':
                    continue
                enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
                if 'customer' not in s:
                    assert len(enabled_actions) == 1, f'More than one action for non-user state {s}' 
                d_sa_strat[(n1,n2)][s] = {a : m.addVar(ub=1.0, lb = -1.0, name=f'Diversity-Dist state {s} action {a} ({n1}, {n2})') for a in enabled_actions}
                d_sa_strat_abs[(n1,n2)][s] = {a : m.addVar(ub=1.0, name=f'Abs diversity-dist state {s} action {a} ({n1}, {n2})', lb = 0) for a in enabled_actions}
                for a in enabled_actions:
                    m.addConstr(d_sa_strat[(n1,n2)][s][a] == s1[s][a] - s2[s][a])
                    m.addConstr(d_sa_strat_abs[(n1,n2)][s][a] == gp.abs_(d_sa_strat[(n1,n2)][s][a]))
            m.addConstr(d_1_visits[i,j] * (1 + sum([0.5 * sum(d_sa_strat_abs[(n1,n2)][s].values()) for s in d_sa_strat_abs[(n1,n2)]])) == 1 )
    

    # d_sa_strat = {}
    # d_sa_strat_abs = {}
    # for s in model.nodes:
    #     if 'positive' in s or 'negative' in s:
    #         continue
    #     if model.edges[list(model.edges(s))[0]]['controllable'] or model.edges[list(model.edges(s))[0]]['action'] == 'env':
    #         continue
    #     enabled_actions = set([model.edges[e]['action'] for e in model.edges(s)])
    #     if 'customer' not in s:
    #         assert len(enabled_actions) == 1, f'More than one action for non-user state {s}' 
    #     d_sa_strat[s] = {a : m.addVar(ub=1.0, lb = -1.0, name=f'Diversity-Dist state {s} action {a}') for a in enabled_actions}
    #     d_sa_strat_abs[s] = {a : m.addVar(ub=1.0, name=f'Abs diversity-dist state {s} action {a}', lb = 0) for a in enabled_actions}
    #     for a in enabled_actions:
    #         m.addConstr(d_sa_strat[s][a] == p_sa[s][a] - solution.strategy[s][a])
    #         m.addConstr(d_sa_strat_abs[s][a] == gp.abs_(d_sa_strat[s][a]))
    # m.addConstr(d_1_visits == sum([0.5 * sum(d_sa_strat_abs[s].values()) for s in d_sa_strat_abs]))
    
    # add small perturbation
    for i in range(len(list_of_strategies)):
        d_1_visits[i,i] == random.uniform(0, 0.00001)
    
    m.setObjective(d_0 + d_1 + d_inf - det(d_1_visits), sense = GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.TIME_LIMIT:
        if m.SolCount == 0:
            return_result = Result(m.Runtime, m.ObjVal, target_prob, {}, timeout, m.MIPGap, m.status)
            m.dispose()
            return return_result
        else:
            return_result = Result(m.Runtime, m.ObjVal, target_prob, {}, timeout, m.MIPGap, GRB.SUBOPTIMAL)
            m.dispose()
            return return_result
        
    assert m.status == GRB.OPTIMAL
    
    strategy = construct_strategy_from_solution(model, p_sa, gurobi_access)
    
    print("Distances strategy")
    print("d_inf", d_inf.X)
    print("d_1", d_1.X)
    print("d_0", d_0.X)
        
    if debug:
        print('Constructed solution')
        print(strategy)
    
    if debug:
        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")
    print(f"Obj: {m.ObjVal:g}")
    
    print("Strategy diff")
    strategy_diff(user_strategy, strategy)
    print("end")
    # print('previous')
    # strategy_diff(user_strategy, solution.strategy)
    # print('both:')
    # strategy_diff(strategy, solution.strategy)
    
    # print('visits')
    # print(expected_visiting_time(model, strategy))
    # print(expected_visiting_time(model, solution.strategy))
    
    return_result = Result(m.Runtime, d_0.X + d_1.X + d_inf.X , target_prob, strategy, timeout, m.MIPGap, m.status)
    m.dispose()
    return return_result

def quadratic_program(model : nx.DiGraph, target_prob : float, user_strategy : dict, timeout = 60*60, threads = 1, debug = False):
    env = gp.Env()
    m = gp.Model("qp", env=env)
    m.setParam('TimeLimit', timeout)
    m.setParam('SoftMemLimit', 2.5)
    m.setParam('Threads', threads)
    
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
        d_sa[s] = {a : m.addVar(ub=1.0, lb = 0, name=f'Abs dist state {s} action {a}') for a in enabled_actions}
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
        return_result = Result(m.Runtime, -0.2, target_prob, {}, timeout, 0, m.status)
        m.dispose()
        return return_result
    
    if m.status == GRB.TIME_LIMIT:
        if m.SolCount == 0:
            return_result = Result(m.Runtime, m.ObjVal, target_prob, {}, timeout, m.MIPGap, m.status)
            m.dispose()
            return return_result
        else:
            return_result = Result(m.Runtime, m.ObjVal, target_prob, {}, timeout, m.MIPGap, GRB.SUBOPTIMAL)
            m.dispose()
            return return_result
     
    assert m.status == GRB.OPTIMAL, f'Status is {m.status}'
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
    
    return_result = Result(m.Runtime, m.ObjVal, target_prob, strategy, timeout, m.MIPGap, m.status)
    m.dispose()
    return return_result


        
def construct_user_strategy(model : nx.DiGraph):
    user_strategy = {}
    
    for s in model.nodes:
        if "positive" in s or "negative" in s:
            continue
        if not model.edges[list(model.edges(s))[0]]['controllable'] and model.edges[list(model.edges(s))[0]]['action'] != 'env':
            enabled_actions = list(set([model.edges[e]['action'] for e in model.edges(s)]))
            distr = random.choices(range(1, 1000), k=len(enabled_actions))
            user_strategy[s] = {enabled_actions[i] : distr[i] / sum(distr) for i in range(len(enabled_actions))}
    
    return user_strategy

def strategy_diff(strat1 : dict, strat2 : dict):
    assert strat1.keys() == strat2.keys()
    for s in strat1:
        assert s in strat2
        assert strat1[s].keys() == strat2[s].keys()
        for a in strat1[s]:
            if round(strat1[s][a], 2) != round(strat2[s][a], 2):
                print(f'In state {s} action {a} differs, {strat1[s][a]} != {strat2[s][a]}')

def textual_strategy(old_strategy : dict, new_strategy : dict):
    assert old_strategy.keys() == new_strategy.keys()
    for s in old_strategy:
        assert s in new_strategy
        assert old_strategy[s].keys() == new_strategy[s].keys()
        printed = False
        for a in old_strategy[s]:
            if round(old_strategy[s][a], 2) != round(new_strategy[s][a], 2) and new_strategy[s][a] != 0:
                if not printed:
                    print(f'In state `{s.replace("customer", "")}\' ')
                    printed = True
                if new_strategy[s][a] > old_strategy[s][a]:
                    print(f'    increase probability of action `{a}\' to {round(new_strategy[s][a], 2)}')
                else:
                    print(f'    decrease probability of action `{a}\' to {round(new_strategy[s][a], 2)}')

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
        n.attr['label'] = n.split(':')[0]
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

def round_probabilities_model(model, precision):
    for s in model.nodes:
        total_sum = sum([round(float(model.edges[e]['prob_weight']), precision) for e in model.edges(s)])
        for e in model.edges(s):
            model.edges[e]['prob_weight'] = round(float(model.edges[e]['prob_weight']), precision) / total_sum
    return model

def round_probabilities_strategy(strategy, precision = 2):
    for s in strategy:
        total_sum = sum(round(strategy[s][a], precision) for a in strategy[s])
        for a in strategy[s]:
            strategy[s][a] = round(strategy[s][a], precision) / total_sum

def run_experiment(param):
    if not args:
        diversity_runs = 4
    else:
        diversity_runs = args.diversity_runs
    
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
    
    print(f'Call {model_path} with reachability probability {p} on strategy {path}')
    import solver
    r_qp = quadratic_program(model, p, user_strategy, timeout=timeout, debug=False)
    r_qp_new =  solver.QuadraticProblem(model, p, user_strategy, timeout=timeout, debug=False).solve()
    o, strat = minimum_reachability(model)
    
    # diversity run
    df_results_div = r_qp.df()
    df_results_div['id'] = 0
    df_results_div['path'] = path
    df_results_div['unknown_fraction'] = 1
    df_results_div['value'] = abs(r_qp.value - r_qp_new.value)
    
    if r_qp.status != GRB.OPTIMAL:
        return pd.DataFrame()
    
    results_div = [r_qp]
    for i in range(diversity_runs):
        print(f'strat {i}')
        r_div = diversity_program_strategy(model, p, user_strategy, results_div, timeout=args.timeout, debug=False)
        r_div_new = solver.QuadraticProblem(model, p, user_strategy, timeout=timeout, debug=False).solve_diverse(results_div)
        if r_div.status != GRB.OPTIMAL:
            continue
        previously_chosen_actions = get_chosen_state_action(user_strategy, results_div)
        chosen_actions = get_chosen_state_action(user_strategy, [r_div])
        unknown_fraction = len([a for a in chosen_actions if a not in previously_chosen_actions]) / len(chosen_actions)
        results_div.append(r_div)
        
        new_df = r_div.df()
        new_df['value'] = abs(r_div.value - r_div_new.value)
        new_df['id'] = i+1
        new_df['path'] = path
        new_df['unknown_fraction'] = unknown_fraction
        df_results_div = pd.concat([df_results_div, new_df])

    return df_results_div

def get_chosen_state_action(user_strategy : dict, results : list):
    chosen_actions = set()
    for r in [r.strategy for r in results]:
        assert user_strategy.keys() == r.keys()
        for s in user_strategy:
            assert s in r
            assert user_strategy[s].keys() == r[s].keys()
            for a in user_strategy[s]:
                if round(user_strategy[s][a], 2) != round(r[s][a], 2):
                    chosen_actions.add((s,a))
    
    return chosen_actions

def manual_execution():
    # manual tests
    with open('out/models/model_bpic12_model-it_4.pickle', 'rb') as handle: #open(f'out/models/model_{name}.pickle', 'rb') as handle:
        model = pickle.load(handle)
    with open('out/user_strategies/model_bpic12_model-it_4_it_9.pickle', 'rb') as handle:
        user_strategy = pickle.load(handle)
        # user_strategy = pickle.load(handle)   
    # print("search_bounds", search_bounds(model, user_strategy))
    
    o, strat = minimum_reachability(model)
    print("optimal", o)
    # r_qp = z3_feasible(model, 0.35, user_strategy, 1, timeout=args.timeout, debug=False)
    r_qp = quadratic_program(model, 0.35, user_strategy, timeout=args.timeout, debug=False)
    results_div = [r_qp]
    df_results_div = r_qp.df()
    df_results_div['id'] = 0
    df_results_div['unknown_fraction'] = 1
    
    for i in range(5):
        print(f'strat {i}')
        r_div = diversity_program_strategy(model, 0.35, user_strategy, results_div, timeout=args.timeout, debug=False)
        previously_chosen_actions = get_chosen_state_action(user_strategy, results_div)
        chosen_actions = get_chosen_state_action(user_strategy, [r_div])
        print("previsouly chosen", previously_chosen_actions)
        print("chosen", chosen_actions)
        unknown_fraction = len([a for a in chosen_actions if a not in previously_chosen_actions]) / len(chosen_actions)
        results_div.append(r_div)
        
        new_df = r_div.df()
        new_df['id'] = i+1
        new_df['path'] = "path"
        new_df['unknown_fraction'] = unknown_fraction
        df_results_div = pd.concat([df_results_div, new_df])
        df_results_div.to_csv("out/results_div.csv")
    
    print()
    print("Opt")
    strategy_diff(user_strategy, r_qp.strategy)
    for r in results_div:
        strategy_diff(user_strategy, r.strategy)
        print()
    
    
    # run_experiment((path, 0.35, args.timeout))
    assert(False)
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(
                    prog = 'benchmarks',
                    description = "File to trigger benchmarks for CE generation in MDP's")
    parser.add_argument('-t', '--timeout', help = "Timeout for program solution", type=int, default = 60*60) 
    parser.add_argument('-s', '--steps', help = "Number of steps for each model", type=int, default = 1)
    parser.add_argument('-i', '--iterations', help = "Iterations for each step", type=int, default = 1)
    parser.add_argument('-c', '--cores', help = "Cores to use to parallelize experiments", type=int, default = 1)
    parser.add_argument('-e', '--experiments', help = "Start profile to filter on", nargs='+', type=str, default = ['greps', 'bpic12', 'bpic17-before', 'bpic17-after', 'bpic17-both', 'spotify'])
    parser.add_argument('-rm', '--rebuild_models', help = "Rebuild models, implies rebuilding models", action = 'store_true')
    parser.add_argument('-rs', '--rebuild_strategies', help = "Rebuild strategies", action = 'store_true')
    parser.add_argument('-mi', '--model_iterations', help = "Number of models to generate for each setting", type=int, default = 10)
    parser.add_argument('-as', '--all_spotify', help = "All spotify models in steps of 100 are generated", action = 'store_true')
    parser.add_argument('-d', '--diversity_runs', help = "Number of diverse counterfactuals", type=int, default = 0)
    args = parser.parse_args()
    
    
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
    
    # trigger single, manual execution
    # manual_execution()
    
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
            model = pickle.load(handle)
        with open(e, 'rb') as handle:
            user_strategy = pickle.load(handle)
        bounds = (0.1,0.5)#search_bounds(model, user_strategy)
        print(bounds)
        experiments.extend([(e, round(bounds[0] + (bounds[1] - bounds[0]) * 1/(args.steps) * s, 4), args.timeout) for s in range(args.steps+1)])
        #experiments.append((e, 1, args.timeout))
    # experiments = [(p, 1/(args.steps)*s, args.timeout) for p in benchmark_strategies for s in range(args.steps+1)]
    
    df_results = pd.DataFrame()
    stored_results = []
    with multiprocessing.Pool(processes=args.cores) as pool:
        result = pool.imap_unordered(run_experiment, experiments)
        for r in result:
            stored_results.append(r)
            df_results = pd.concat([df_results, r])
            df_results.to_csv("out/results_div.csv")
    # result = [run_experiment_diverse(e) for e in experiments]
    result = stored_results
    print("Done")