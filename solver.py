import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from Result import Result
from benchmarks import strategy_diff
import random
random.seed(42)

class QuadraticProblem:
    
    def construct_strategy_from_solution(model : nx.DiGraph, p_sa : dict, user_strategy = {}):
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
                strategy[s] = {a : p_sa[s][a].X for a in enabled_actions}
        
        return strategy

    def __init__(self, model : nx.DiGraph, target_prob : float, user_strategy : dict, timeout = 60*60, threads = 1, debug = False):   
        self.env = gp.Env()
        self.m = gp.Model("qp", env=self.env)
        self.m.setParam('TimeLimit', timeout)
        self.m.setParam('SoftMemLimit', 2.5)
        self.m.setParam('Threads', threads)
        
        self.model = model
        self.target_prob = target_prob
        self.user_strategy = user_strategy
        self.timeout = timeout
        self.debug = debug
        
        self.p = {s : self.m.addVar(ub=1.0, name=s, lb = 0) for s in self.model.nodes}
        self.p_sa = {}
        self.d_sa = {}
        
        self.d_0 = self.m.addVar(name='d0', lb = 0)
        self.d_1 = self.m.addVar(name='d1', lb = 0, ub=1)
        self.d_inf = self.m.addVar(name='d_inf', lb = 0, ub=1)
        
        start_state = [s for s in self.model.nodes if 'q0: start' in s]
        assert len(start_state) == 1, start_state
        self.start_state = start_state[0]
        
        target_state = [s for s in self.model.nodes if 'negative' in s]
        assert len(target_state) == 1, target_state
        self.target_state = target_state[0]
                
        self.reaching_states = [s for s in model.nodes if nx.has_path(model, s, self.target_state)]

        self.encode_actions()
        self.encode_model()
        self.reachability_constraint()
        
        self.strict_proximal()
        # relaxed proximal
        # use d_sa to encode d_1 norm
        self.m.addConstr(self.d_1 == sum([0.5 * sum(self.d_sa[s].values()) for s in self.d_sa]) / len(self.user_strategy))
        self.encode_sparsity()
        
    def encode_actions(self) -> dict:
        # encode actions
        for s in self.model.nodes:
            # Encode pos and neg states as absorbing, i.e. without available actions. Thus, the can not be included in p_sa
            if 'positive' in s:
                assert s not in self.reaching_states
                self.m.addConstr(self.p[s] == 0)
                continue
            if 'negative' in s:
                self.m.addConstr(self.p[s] == 1)
                assert self.target_state == s
                continue
            enabled_actions = set([self.model.edges[e]['action'] for e in self.model.edges(s)])
            if 'customer' not in s:
                assert len(enabled_actions) <= 1, f'More than one action for non-user state {s} : {enabled_actions}' 
            self.p_sa[s] = {a : self.m.addVar(ub=1.0, name=s+'_'+a, lb = 0) for a in enabled_actions}
            self.m.addConstr(sum(list(self.p_sa[s].values())) == 1) # scheduler sums up to one
            for a in enabled_actions:
                self.m.addConstr(self.p_sa[s][a] <= 1)
            
    def encode_model(self):
        # encode model
        for s in self.p_sa:
            enabled_actions = set([self.model.edges[e]['action'] for e in self.model.edges(s)])
            assert len(enabled_actions) >= 1, f'State{s} has no enabled action'
            if s in self.reaching_states:
                self.m.addConstr(self.p[s] == sum([self.p_sa[s][self.model.edges[e]['action']] * float(self.model.edges[e]['prob_weight']) * self.p[e[1]] for e in self.model.edges(s)]))
            else:
                # Not reachable states are still in strategy
                if self.debug:
                    print(f'Set {s} to 0')
                self.m.addConstr(self.p[s] == 0)

    def reachability_constraint(self):
        # encode reachability constraint
        self.m.addConstr(self.p[self.start_state] <= self.target_prob)
    
    def strict_proximal(self):
        # strict proximal
        def add_abs(var, prob, constr):
            self.m.addConstr(prob - constr <= var)
            self.m.addConstr(constr - prob <= var)
            
        for s in self.model.nodes:
            if 'positive' in s or 'negative' in s:
                continue
            if self.model.edges[list(self.model.edges(s))[0]]['controllable'] or self.model.edges[list(self.model.edges(s))[0]]['action'] == 'env':
                continue
            enabled_actions = set([self.model.edges[e]['action'] for e in self.model.edges(s)])
            if 'customer' not in s:
                assert len(enabled_actions) == 1, f'More than one action for non-user state {s}' 
            self.d_sa[s] = {a : self.m.addVar(ub=1.0, lb = 0, name=f'Abs dist state {s} action {a}') for a in enabled_actions}
            for a in enabled_actions:
                add_abs(self.d_sa[s][a], self.p_sa[s][a], self.user_strategy[s][a])
            # encode d_inf constraint
            self.m.addConstr(0.5 * sum(self.d_sa[s].values()) <= self.d_inf)
    
    def encode_sparsity(self):
        # encode sparsity
        decision_changed = {}
        dist_binary = {}
        for s in self.d_sa:
            dist_binary[s] = self.m.addVar(ub=1.0, name=f'State {s} was changed', lb = 0, vtype=gp.GRB.BINARY)
            decision_changed[s] = self.m.addVar(ub=1.0, name=f'Var dist state {s}', lb = 0)
            self.m.addConstr(decision_changed[s] == 0.5 * sum(self.d_sa[s].values()))
            self.m.addConstr(decision_changed[s] <= 10 * dist_binary[s])
        self.m.addConstr(sum(dist_binary.values()) == self.d_0)
        
    def get_solution(self):
        if self.m.status == GRB.INFEASIBLE:
            return Result(self.m.Runtime, -0.2, self.target_prob, {}, self.timeout, 0, self.m.status)
        
        # compute result as in diverse target function include determinant
        if self.m.status == GRB.TIME_LIMIT:
            if self.m.SolCount == 0:
                return Result(self.m.Runtime, 0, self.target_prob, {}, self.timeout, self.m.MIPGap, self.m.status)
            else:
                strategy = QuadraticProblem.construct_strategy_from_solution(self.model, self.p_sa)
                return Result(self.m.Runtime, self.d_0.X + self.d_1.X + self.d_inf.X, self.target_prob, strategy, self.timeout, self.m.MIPGap, GRB.SUBOPTIMAL)
            
        strategy = QuadraticProblem.construct_strategy_from_solution(self.model, self.p_sa)
        return Result(self.m.Runtime, self.d_0.X + self.d_1.X + self.d_inf.X, self.target_prob, strategy, self.timeout, self.m.MIPGap, self.m.status)
            
    def solve(self):       
        self.m.setObjective(self.d_0 + self.d_1 + self.d_inf, sense = GRB.MINIMIZE)
        self.m.optimize()
        
        return_result = self.get_solution()
        if self.m.status == GRB.INFEASIBLE or self.m.status == GRB.TIME_LIMIT:
            self.m.dispose()
            return return_result
        
        assert self.m.status == GRB.OPTIMAL, f'Status is {self.m.status}'
        print("Distances")
        print("d_inf", self.d_inf.X)
        print("d_1", self.d_1.X)
        print("d_0", self.d_0.X)
        if self.debug:
            for v in self.m.getVars():
                print(f"{v.VarName} {v.X:g}")
            print(f"Obj: {self.m.ObjVal:g}")
        
        strategy = QuadraticProblem.construct_strategy_from_solution(self.model, self.p_sa)
        if self.debug:
            print('Constructed solution')
            print(strategy)
        
        strategy_diff(self.user_strategy, strategy)
        
        self.m.dispose()
        return return_result
    
    def solve_diverse(self, solutions : list):
        list_of_strategies = [s.strategy for s in solutions]
        list_of_strategies.insert(0, self.p_sa)
        names_list_of_strategies = [f's{i}' for i in range(len(list_of_strategies)-1)]
        names_list_of_strategies.insert(0, "p_sa")
        
        # Encode diversity pairwise
        
        def det(A):
            v = self.m.addVar(lb=-float("inf"))
            if A.shape == (2,2):
                self.m.addConstr(v == A[0,0]*A[1,1]-A[1,0]*A[0,1])
                return v
            # if A.shape == (3,3):
            #     self.m.addGenConstrNL(v, A[0,0]*A[1,1]*A[2,2] + A[0,1]*A[1,2]*A[2,0] + A[0,2]*A[1,0]*A[2,1] - A[0,2]*A[1,1]*A[2,0] - A[0,1]*A[1,0]*A[2,2] - A[0,0]*A[1,2]*A[2,1])
            #     return v
            expr = gp.QuadExpr()
            cofactor = 1
            for i in range(A.shape[1]):
                cols = [c for c in range(A.shape[1]) if c != i]
                expr += cofactor*A[0,i]*det(A[1:][:,cols])
                cofactor = -cofactor
            self.m.addConstr(v == expr)
            return v
        
        d_sa_strat = {}
        d_sa_strat_abs = {}
        d_1_visits = self.m.addMVar((len(list_of_strategies),len(list_of_strategies)), lb=-float("inf"))
        for i in range(len(list_of_strategies)):
            s1 = list_of_strategies[i]
            n1 = names_list_of_strategies[i]
            for j in range(len(list_of_strategies)):
                s2 = list_of_strategies[j]
                n2 = names_list_of_strategies[j]
                d_sa_strat[(n1,n2)] = {}
                d_sa_strat_abs[(n1,n2)] = {}
                for s in self.model.nodes:
                    if 'positive' in s or 'negative' in s:
                        continue
                    if self.model.edges[list(self.model.edges(s))[0]]['controllable'] or self.model.edges[list(self.model.edges(s))[0]]['action'] == 'env':
                        continue
                    enabled_actions = set([self.model.edges[e]['action'] for e in self.model.edges(s)])
                    if 'customer' not in s:
                        assert len(enabled_actions) == 1, f'More than one action for non-user state {s}' 
                    d_sa_strat[(n1,n2)][s] = {a : self.m.addVar(ub=1.0, lb = -1.0, name=f'Diversity-Dist state {s} action {a} ({n1}, {n2})') for a in enabled_actions}
                    d_sa_strat_abs[(n1,n2)][s] = {a : self.m.addVar(ub=1.0, name=f'Abs diversity-dist state {s} action {a} ({n1}, {n2})', lb = 0) for a in enabled_actions}
                    for a in enabled_actions:
                        self.m.addConstr(d_sa_strat[(n1,n2)][s][a] == s1[s][a] - s2[s][a])
                        self.m.addConstr(d_sa_strat_abs[(n1,n2)][s][a] == gp.abs_(d_sa_strat[(n1,n2)][s][a]))
                self.m.addConstr(d_1_visits[i,j] * (1 + sum([0.5 * sum(d_sa_strat_abs[(n1,n2)][s].values()) for s in d_sa_strat_abs[(n1,n2)]])) == 1 )

        # add small perturbation
        for i in range(len(list_of_strategies)):
            d_1_visits[i,i] == random.uniform(0, 0.00001)
        
        
        self.m.setObjective(self.d_0 + self.d_1 + self.d_inf - len(solutions) * det(d_1_visits), sense = GRB.MINIMIZE)
        self.m.optimize()
        
        # process solution
        return_result = self.get_solution()
        if self.m.status == GRB.INFEASIBLE or self.m.status == GRB.TIME_LIMIT:
            self.m.dispose()
            return return_result
        
        assert self.m.status == GRB.OPTIMAL, f'Status is {self.m.status}'
        print("Distances function")
        print("d_inf", self.d_inf.X)
        print("d_1", self.d_1.X)
        print("d_0", self.d_0.X)
        if self.debug:
            for v in self.m.getVars():
                print(f"{v.VarName} {v.X:g}")
            print(f"Obj: {self.m.ObjVal:g}")
        
        strategy = QuadraticProblem.construct_strategy_from_solution(self.model, self.p_sa)
        if self.debug:
            print('Constructed solution')
            print(strategy)
        
        strategy_diff(self.user_strategy, strategy)
        
        self.m.dispose()
        return return_result