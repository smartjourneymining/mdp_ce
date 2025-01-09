"""
 Functions were used for geometric programming solution and copied for later here.
 Contain all geometric programming functionality
 IMPORTANT: Worked only in benchmarks.py file
"""
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


def strategy_slack(strategy):
    total = 0
    for s in strategy:
        total += 1 - sum(strategy[s][a] for a in strategy[s])
    return total 