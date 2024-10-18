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


m.addConstr(p5_action_Quit + p5_action_ReSubmit <= 1)
m.addConstr(p2_action_Application + p2_action_Quit <= 1)
m.addConstr(p1_action_Application + p1_action_Quit <= 1)
m.addConstr(p0_action_Application + p0_action_Consult <= 1)

m.addConstr(places[8] == 1) # unreachable end receives 0
#m.addConstr(places[7] == 0) # state itself receives 1
m.addConstr(places[6] >= 0.8 * places[7] + 0.2 * places[8])
m.addConstr(places[5] >= p5_action_Quit * places[8] + p5_action_ReSubmit * places[6])
m.addConstr(places[4] >= 0.1 * places[5] + 0.9 * places[7])
m.addConstr(places[3] >= 0.5 * places[5] + 0.5 * places[7])
m.addConstr(places[2] >= p2_action_Quit * places[8] + p2_action_Application * places[4] )
m.addConstr(places[1] >= p1_action_Quit * places[8] + p1_action_Application * places[2])
m.addConstr(places[0] >= p0_action_Application * 0.95 * places[3] + p0_action_Application * 0.05 * places[1] + p0_action_Consult * places[2])

m.addConstr(places[0] <= 0.4)

e = m.addVar(name='epsilon', lb = 0, ub=1)

# # strict proximal
# m.addConstr(e >= 1 - p0_action_Application)
# m.addConstr(e >= p0_action_Consult)
# m.addConstr(e >= (0.8 - p1_action_Quit))
# m.addConstr(-e <= 0.8 - p1_action_Quit)
# m.addConstr(e >= (0.2 - p1_action_Application))
# m.addConstr(-e <= (0.2 - p1_action_Application))
# m.addConstr(e >= 1 - p2_action_Quit)
# m.addConstr(-e <= 1 - p2_action_Quit)
# m.addConstr(e >= p2_action_Application)
# m.addConstr(-e <= p2_action_Application)
# m.addConstr(e >= (0.7 - p5_action_Quit))
# m.addConstr(-e <= (0.7 - p5_action_Quit))
# m.addConstr(e >= (0.3 - p5_action_ReSubmit))
# m.addConstr(-e <= (0.3 - p5_action_ReSubmit))

# Set objective
obj = e
m.setObjective(sum([v for v in all_vars]), sense = GRB.MAXIMIZE)

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
