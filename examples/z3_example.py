
from z3 import *

def abs(x):
    return If(x >= 0,x,-x)

# all_actions = ["Error", "Consult", "Application", "Application1", "Review", "ReSubmit", "Reject", "Grant"]

places = [ Real('p%s' % i) for i in range(9) ]

# places_actions = []
# for i in range(9):
#     places_actions.append([ Real('p%s_action:%s' % (i,j)) for j in all_actions ])

s = Solver()
# s = Optimize()

# for p in places_actions:
#     for a in p:
#         s.add(a >= 0, a <= 1)
        
p5_action_Quit = Real('p5_action_Quit')
p5_action_ReSubmit = Real('p5_action_ReSubmit')
p2_action_Quit = Real('p2_action_Quit')
p2_action_Application = Real('p2_action_Application')
p1_action_Quit = Real('p1_action_Quit')
p1_action_Application = Real('p1_action_Application')
p0_action_Application = Real('p0_action_Application')
p0_action_Consult = Real('p0_action_Consult')

all_vars = [p5_action_Quit, p5_action_ReSubmit, p2_action_Quit, p2_action_Application, p1_action_Quit, p1_action_Application, p0_action_Consult, p0_action_Application]

for v in all_vars:
    s.add(v <= 1)
    s.add(v >= 0)
    
s.add(p5_action_Quit + p5_action_ReSubmit == 1)
s.add(p2_action_Application + p2_action_Quit == 1)
s.add(p1_action_Application + p1_action_Quit == 1)
s.add(p0_action_Application + p0_action_Consult == 1)

s.add(places[8] == 1) # unreachable end receives 0
s.add(places[7] == 0) # state itself receives 1
s.add(places[6] == 0.8 * places[7] + 0.2 * places[8])
s.add(places[5] == p5_action_Quit * places[8] + p5_action_ReSubmit * places[6])
s.add(places[4] == 0.1 * places[5] + 0.9 * places[7])
s.add(places[3] == 0.5 * places[5] + 0.5 * places[7])
s.add(places[2] == p2_action_Quit * places[8] + p2_action_Application * places[4] )
s.add(places[1] == p1_action_Quit * places[8] + p1_action_Application * places[2])
s.add(places[0] == p0_action_Application * 0.95 * places[3] + p0_action_Application * 0.05 * places[1] + p0_action_Consult * places[2])

s.add(places[0] <= 0.2)
# todo adjust for available actions

e = Real('epsilon')

# strict proximal
# s.add(e >= 1 - p0_action_Application)
# s.add(e >= p0_action_Consult)
# s.add(e**2 >= (0.8 - p1_action_Quit)**2)
# s.add(e**2 >= (0.2 - p1_action_Application)**2)
# s.add(e**2 >= 1 - p2_action_Quit)
# s.add(e**2 >= p2_action_Application)
# s.add(e**2 >= (0.7 - p5_action_Quit)**2)
# s.add(e**2 >= (0.3 - p5_action_ReSubmit)**2)

s.add(e >= abs(1 - p0_action_Application))
s.add(e >= abs(0.8 - p1_action_Quit))
s.add(e >= abs(0.2 - p1_action_Application))
s.add(e >= abs(1 - p2_action_Quit))
s.add(e >= abs(p2_action_Application))
s.add(e >= abs(0.7 - p5_action_Quit))
s.add(e >= abs(0.3 - p5_action_ReSubmit))

# relaxed proximal
# h1 = Real('h1')
# s.add(h1 == (abs(1 - p0_action_Application) + abs(0 - p0_action_Consult))/2)
# h2 = Real('h2')
# s.add(h2 == (abs(0.8 - p1_action_Quit) + abs(0.2 - p1_action_Application))/2)
# h3 = Real('h3')
# s.add(h3 == (abs(1 - p2_action_Quit) + abs(0 - p2_action_Application))/2)
# h4 = Real('h4')
# s.add(h4 == (abs(0.7 - p5_action_Quit) + abs(0.3 - p5_action_ReSubmit))/2)

# h5 = Real('h5')
# s.add(h5 == (h1 + h2 + h3 + h4 ) / 4)
# s.add(e == h5)

# add sparsity
# s.add(Or([Sum([h1, h2, h3, h4]) == h1, Sum([h1, h2, h3, h4]) == h2, Sum([h1, h2, h3, h4]) == h3, Sum([h1, h2, h3, h4]) == h4]))

s.add(e >= 0)
s.add(e <= 0.5094)

# h_e = Int('feasible-eps')
# s.add(h_e <= 100)
# s.add(h_e >= 0)
# s.add(h_e == e * 100)
# #s.add(h_e == 15)
# s.minimize(h_e)

print(s)
print(s.check())

m = s.model()
for i in (sorted ([(d, m[d]) for d in m], key = lambda x: str(x[0]))):
    print(i)
