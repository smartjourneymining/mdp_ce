import cvxpy as cp

def run_program():
    places = [cp.Variable(pos=True, name='p%s' % i) for i in range(9)]

    p5_action_Quit = cp.Variable(pos=True, name='p5_action_Quit')
    p5_action_ReSubmit = cp.Variable(pos=True, name='p5_action_ReSubmit')
    p2_action_Quit = cp.Variable(pos=True, name='p2_action_Quit')
    p2_action_Application = cp.Variable(pos=True, name='p2_action_Application')
    p1_action_Quit = cp.Variable(pos=True, name='p1_action_Quit')
    p1_action_Application = cp.Variable(pos=True, name='p1_action_Application')
    p0_action_Application = cp.Variable(pos=True, name='p0_action_Application')
    p0_action_Consult = cp.Variable(pos=True, name='p0_action_Consult')

    all_vars = [p5_action_Quit, p5_action_ReSubmit, p2_action_Quit, p2_action_Application, p1_action_Quit, p1_action_Application, p0_action_Consult, p0_action_Application]

    constraints = []

    print(cp.installed_solvers())
    
    for v in all_vars:
        constraints.append(v <= 1)
        
    # place constraints    
    constraints.append(p5_action_Quit + (0.3 + p5_action_ReSubmit) <= 1)
    constraints.append((0.00001 + p2_action_Application) + p2_action_Quit <= 1)
    constraints.append((0.2 + p1_action_Application) + p1_action_Quit <= 1)
    constraints.append(p0_action_Application + (0.00001 + p0_action_Consult) <= 1)

    constraints.append(places[8] == 1) # unreachable end receives 0
    #constraints.append(places[7] >= 0) # state itself receives 1
    constraints.append(places[6] >= 0.8 * 0 + 0.2 * places[8]) # places[7] = 0
    constraints.append(places[5] >= p5_action_Quit * places[8] + (0.3 + p5_action_ReSubmit) * places[6])
    constraints.append(places[4] >= 0.1 * places[5] + 0.9 * 0) # places[7] = 0
    constraints.append(places[3] >= 0.5 * places[5] + 0.5 * 0) # places[7] = 0
    constraints.append(places[2] >= p2_action_Quit * places[8] + (0.00001 + p2_action_Application) * places[4])
    constraints.append(places[1] >= p1_action_Quit * places[8] + (0.2 + p1_action_Application) * places[2])
    constraints.append(places[0] >= p0_action_Application * 0.95 * places[3] + p0_action_Application * 0.05 * places[1] + (0.00001 + p0_action_Consult) * places[2])
    
    constraints.append(0.35 >= places[0])
        
    ##############################
    # UNCOMMENT FOR REACHABILITY PROBLEM
    # problem = cp.Problem(cp.Minimize(sum([1/v for v in all_vars])), constraints)
    # print(problem)
    # print("Is this problem DGP?", problem.is_dgp())

    # problem.solve(gp=True, solver="MOSEK", save_file='geometric.msp')
    # print(problem)
    # print("Value:", problem.value)

    # total_sum = 0
    # stoch_sum = 0
    # for v in all_vars:
    #     print(v, v.value)
    #     total_sum += 1/v.value
    #     stoch_sum += v.value
    # print("sum", total_sum)
    # print("stoch sum", stoch_sum)
    
    # for v in places:
    #     print(v, v.value)
        
    # return
    ##############################
    
    # add norm constraints
    d_inf = cp.Variable(pos=True,name='d_inf')

    added_vars = []
    min_vars = []
    max_vars = []
    def encode_abs(var, prob, name):
        d = cp.Variable(pos=True, name=name)
        d_min = cp.Variable(pos=True, name=name+'_min')
        d_max = cp.Variable(pos=True, name=name+'_max')
        
        added_vars.append(d)
        min_vars.append(d_min)
        max_vars.append(d_max)

        constraints.append(d + d_min <= d_max)
        constraints.append(d <= cp.diff_pos(d_max, d_min))

        constraints.append(d_min <= cp.minimum(prob, var))
        # constraints.append(d_min <= prob)
        # constraints.append(d_min <= var)
        constraints.append(d_max >= cp.maximum(prob, var))
        # constraints.append(d_max >= prob)
        # constraints.append(d_max >= var)
        constraints.append(d_max <= 1)
    
    # encode_abs(p0_action_Application, 0.99999, "p0_dist_application")
    # encode_abs(p0_action_Consult, 0.00001, "p0_dist_consult")
    constraints.append(d_inf >=  p0_action_Consult)
    
    # encode_abs(p1_action_Quit, 0.8, "p1_dist_quit")
    # encode_abs(p1_action_Application, 0.2, "p1_dist_application")
    constraints.append(d_inf >= p1_action_Application)
    
    # encode_abs(p2_action_Quit, 0.99999, "p2_dist_quit")
    # encode_abs(p2_action_Application, 0.00001, "p2_dist_application")
    constraints.append(d_inf >= p2_action_Application)
    
    # encode_abs(p5_action_Quit, 0.7, "p5_dist_quit")
    # encode_abs(p5_action_ReSubmit, 0.3, "p5_dist_resubmit")
    constraints.append(d_inf >= p5_action_ReSubmit)
        
    # ensure well-formedness
    for constraint in constraints:
        assert constraint.is_dgp(), constraint
        
    problem = cp.Problem(cp.Minimize(sum([1/v for v in all_vars]) + d_inf), constraints)
    # problem = cp.Problem(cp.Minimize(d_inf), constraints)

    print(problem)
    print("Is this problem DGP?", problem.is_dgp())


    problem.solve(gp=True, solver="MOSEK", save_file='geometric.tsk')
    print("sol", problem.value)

    print("d_inf")
    print(d_inf, d_inf.value)
    print("added_vars:")
    for v in added_vars:
        print(v, v.value)
        
    print("max_vars:")
    for v in max_vars:
        print(v, v.value)
    print("min_vars")
    for v in min_vars:
        print(v, v.value)
        
    print("all_vars:")
    total_sum = 0
    stoch_sum = 0
    p5_action_ReSubmit.value += 0.3
    p2_action_Application.value += 0.00001
    p1_action_Application.value += 0.2
    p0_action_Application.value += 0.00001
    for v in all_vars:
        print(v, v.value)
        total_sum += 1/v.value
        stoch_sum += v.value
    print("sum", total_sum)
    print("stoch sum", stoch_sum)
    
    print(places)
    for v in places:
        print(v, v.value)
if __name__ == '__main__':
    run_program()