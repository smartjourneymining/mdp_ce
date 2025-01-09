def previous_encoding():
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    m = gp.Model("qp")
    #m.Params.NonConvex = 0

    places = [m.addVar(ub=1.0, name='p%s' % i, lb = 0) for i in range(9)]

    p5_action_Quit = m.addVar(ub=1.0, name='p5_action_Quit', lb = 0)
    p5_action_ReSubmit = m.addVar(ub=1.0, name='p5_action_ReSubmit', lb = 0)
    p2_action_Quit = m.addVar(ub=1.0, name='p2_action_Quit', lb = 0)
    p2_action_Application = m.addVar(ub=1.0, name='p2_action_Application', lb = 0)
    p1_action_Quit = m.addVar(ub=1.0, name='p1_action_Quit', lb = 0)
    p1_action_Application = m.addVar(ub=1.0, name='p1_action_Application', lb = 0)
    p0_action_Application = m.addVar(ub=1.0, name='p0_action_Application', lb = 0)
    p0_action_Consult = m.addVar(ub=1.0, name='p0_action_Consult', lb = 0)

    all_vars = [p5_action_Quit, p5_action_ReSubmit, p2_action_Quit, p2_action_Application, p1_action_Quit, p1_action_Application, p0_action_Consult, p0_action_Application]

    for v in all_vars:
        m.addConstr(v <= 1)
        #m.addConstr(v >= 0)


    m.addConstr(p5_action_Quit + p5_action_ReSubmit == 1)
    m.addConstr(p2_action_Application + p2_action_Quit == 1)
    m.addConstr(p1_action_Application + p1_action_Quit == 1)
    m.addConstr(p0_action_Application + p0_action_Consult == 1)

    m.addConstr(places[8] == 1) # unreachable end receives 0
    m.addConstr(places[7] == 0) # state itself receives 1
    m.addConstr(places[6] >= 0.8 * places[7] + 0.2 * places[8])
    m.addConstr(places[5] >= p5_action_Quit * places[8] + p5_action_ReSubmit * places[6])
    m.addConstr(places[4] >= 0.1 * places[5] + 0.9 * places[7])
    m.addConstr(places[3] >= 0.5 * places[5] + 0.5 * places[7])
    m.addConstr(places[2] >= p2_action_Quit * places[8] + p2_action_Application * places[4] )
    m.addConstr(places[1] >= p1_action_Quit * places[8] + p1_action_Application * places[2])
    m.addConstr(places[0] >= p0_action_Application * 0.95 * places[3] + p0_action_Application * 0.05 * places[1] + p0_action_Consult * places[2])

    m.addConstr(places[0] <= 0.2)

    e = m.addVar(name='epsilon', lb = 0, ub=1)

    # strict proximal

    def add_abs(var, prob, constr):
        m.addConstr(prob - var <= constr)
        m.addConstr(-constr <= prob - var)
        
    add_abs(e, 1, p0_action_Application)
    add_abs(e, 0, p0_action_Consult)
    add_abs(e, 0.8, p1_action_Quit)
    add_abs(e, 0.2, p1_action_Application)
    add_abs(e, 1, p2_action_Quit)
    add_abs(e, 0, p2_action_Application)
    add_abs(e, 0.7, p5_action_Quit)
    add_abs(e, 0.3, p5_action_ReSubmit)

    # Set objective
    obj = e
    m.setObjective(e, sense = GRB.MINIMIZE)

    m.optimize()

    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")

    print(f"Obj: {m.ObjVal:g}")

    #x.VType = GRB.INTEGER
    #y.VType = GRB.INTEGER
    #z.VType = GRB.INTEGER
    #    
    #m.optimize()
    #
    #for v in m.getVars():
    #    print(f"{v.VarName} {v.X:g}")
    #
    #print(f"Obj: {m.ObjVal:g}")


import benchmarks
import networkx as nx
import matplotlib.pyplot as plt

model = nx.DiGraph()
model.add_edge("q0: start_customer", "error_customer", action = 'apply', prob_weight = 0.05, controllable=False)
model.add_edge("q0: start_customer", "application", action = 'apply', prob_weight = 0.95, controllable=False)
model.add_edge("q0: start_customer", "consultation_customer", action = 'consult', prob_weight = 1, controllable=False)
model.add_edge("application", "rework_customer", action = 'provider', prob_weight = 0.5, controllable=True)
model.add_edge("application", "positive", action = 'provider', prob_weight = 0.5, controllable=True)
model.add_edge("error_customer", "consultation_customer", action = 'consult', prob_weight = 1, controllable=False)
model.add_edge("error_customer", "negative", action = 'quit', prob_weight = 1, controllable=False)
model.add_edge("consultation_customer", "application+", action = 'apply', prob_weight = 1, controllable=False)
model.add_edge("consultation_customer", "negative", action = 'quit', prob_weight = 1, controllable=False)
model.add_edge("rework_customer", "resubmit", action = 'submit', prob_weight = 1, controllable=False)
model.add_edge("rework_customer", "negative", action = 'quit', prob_weight = 1, controllable=False)
model.add_edge("resubmit", "positive", action = 'provider', prob_weight = 0.8, controllable=True)
model.add_edge("resubmit", "negative", action = 'provider', prob_weight = 0.2, controllable=True)
model.add_edge("application+", "positive", action = 'provider', prob_weight = 0.9, controllable=True)
model.add_edge("application+", "rework_customer", action = 'provider', prob_weight = 0.1, controllable=True)

user_strategy = {}
user_strategy['q0: start_customer'] = {'apply': 1, 'consult' : 0}
user_strategy['error_customer'] = {'consult': 0.2, 'quit' : 0.8}
user_strategy['consultation_customer'] = {'quit': 1, 'apply' : 0}
user_strategy['rework_customer'] = {'submit': 0.3, 'quit' : 0.7}

print("Reachability so far:")
print(benchmarks.evaluate_strategy(model, user_strategy))
print("#######################")
print()

result = benchmarks.quadratic_program(model, 0.2, user_strategy)
benchmarks.plot_changes(model, "example", user_strategy=user_strategy, counterfactual_strategy=result.strategy, layout="dot")
print("#######################")
print()

result_div = benchmarks.diversity_program(model, 0.2, user_strategy, result)
