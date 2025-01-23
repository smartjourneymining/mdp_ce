import copy 
import networkx as nx
import subprocess
import pandas as pd
from os import stat

def get_states(g_before, g_after):
    states_set = set()
    for s in g_before.nodes:
        states_set.add(s)
    for s in g_after.nodes:
        states_set.add(s)
    states_set = list(states_set)
    states_set.sort()
    states = {}
    for n,i in zip(list(states_set),range(1,len(states_set)+1)): # reserve 0 for start state
        if n == "start":
            continue
        states[n] = str(i)
    states["start"] = "0"
    return states

class PrismPrinter:
    def __init__(self, g, path, name):
        # format state names to prism requirements and rename actions
        g = copy.deepcopy(g)
        new_state_names = {s : s.replace(" ", "").replace (":", "").replace("-", "") for s in g}
        g = nx.relabel_nodes(g, new_state_names)
        for e in g.edges:
            if 'action' in g.edges[e]:
                g.edges[e]['action'] = g.edges[e]['action'].replace(" ", "").replace (":", "").replace("-", "")
        self.g = g
        self.path = path
        self.terminal = [s for s in g if "neg" in s or "pos" in s]
        self.f = open(path+name, "w+")

        # can construct new state ids, in alergia can several states with same name exist (unequal to PM approach)
        states = {}
        for n,i in zip(list(g.nodes),range(1,len(g.nodes)+1)): # reserve 0 for start state
            if "q0start" in n:
                states[n] = "0"
                continue
            states[n] = str(i)
        self.states = states

        self.states_max = max([int(self.states[x]) for x in self.states])+10000

    def __del__(self):
        self.f.close()

    def get_outgoing_edges(self, edges : list, debug = False):
        """Computes a mapping for list of edges that computes a mapping of states to activities ands their corresponding possible outcome states.

        Args:
            edges (list): List of edges to compute outgoing edge-activities for
            debug (bool, optional): Debug option to print additional information. Defaults to False.

        Returns:
            dict: return dict mapping {states -> {actions -> [outcome_states]}}
        """
        outgoing_edges = {}
        for e in edges:
            if debug:
                print(e, self.g.edges[e])
                print()
            action = self.g.edges[e]['action']
            if e[0] not in outgoing_edges:
                outgoing_edges[e[0]] = {}
            if action not in outgoing_edges[e[0]]:
                outgoing_edges[e[0]][action] = [e]
            else:
                outgoing_edges[e[0]][action].append(e)
        return outgoing_edges
    
    def write_edges(self, outgoing_edges:list, reward_dicts : dict, write_attributes = False, steps_max = 50, write_parameterized = False, max_gas = 1000, min_gas = -1000):
        """Writes edges to self.f and updates the reward structure. To not update reward structure, pass dummy structure.

        Args:
            outgoing_edges (list): Edges to write
            reward_dicts (dict): Reward structure to update inplace
        """
        for state in outgoing_edges:
            for action in outgoing_edges[state]:            
                # first edge
                possible_states = outgoing_edges[state][action]

                # substitute states for rewards
                possible_states_substitute = {}
                for sub_state in possible_states:
                    possible_states_substitute[sub_state[1]] = str(self.states_max)
                    self.states_max += 1

                self.f.write('[' + str(state) + 'SEP' + str(action) + '] state=' + self.states[state] + ' ->' )
                for i in range(len(possible_states)):
                    if i != 0:
                        self.f.write(" + ")
                    p = possible_states[i]
                    gas = int(self.g.edges[p]['gas'])
                    # write parameterized environment
                    assert(not write_parameterized or len(possible_states)<= 2) # write_parameterized -> len(possible_states) <= 2
                    if write_parameterized and len(possible_states) == 2: # if len == 1, nothing to parameterize
                        assert("customer" in p[1] or "company" in p[1])
                        # for envprob = 0 : as observed, envprob = 1 : user, envprob = -1 : company
                        if "customer" in p[1]:
                            # self.f.write("(" + str(self.g.edges[p]['prob_weight'])+" + " + str(1-self.g.edges[p]['prob_weight']) + " * (1 - envprob) ) : ")
                            self.f.write("(" + str(1-self.g.edges[p]['prob_weight'])+" * max(envprob, 0) + " + str(self.g.edges[p]['prob_weight']) + " * min(1+envprob,1) ) : ")
                        if "company" in p[1]:
                            #self.f.write("(" + str(self.g.edges[p]['prob_weight'])+" * envprob ) : ")
                            self.f.write("(" + str(1-self.g.edges[p]['prob_weight']) +" * max(-envprob,0) + " + str(self.g.edges[p]['prob_weight']) + " * min(1-envprob,1) ) : ")
                    else:
                        self.f.write(str(self.g.edges[p]['prob_weight'])+" : ")
                    self.f.write("(state'="+ possible_states_substitute[p[1]]+" )"
                                 + ("& (min_gas' = min(min_gas, gas + "+str(gas)+"))" if write_attributes else "" )
                                 + ("& (gas' = " + ("min(" + str(max_gas) if gas > 0 else "max(" + str(min_gas)  ) + ", gas + "+str(gas)+"))" if write_attributes else "") 
                                 + ("& (steps' = min(steps+1," + str(steps_max) + "))" if write_attributes else "" ))
                self.f.write("; \n")

                # transitions for substitute states & update reward_dict structure
                for i in range(len(possible_states)):
                    p = possible_states[i]
                    self.f.write('[' + ']state=' + possible_states_substitute[p[1]] + " -> (state'= " + self.states[p[1]] + ')')
                    reward_dicts['steps'][possible_states_substitute[p[1]]] = 1
                    reward_dicts['gas'][possible_states_substitute[p[1]]] = int(self.g.edges[p]['gas'])
                    if p[1] in self.terminal:
                        self.f.write(" & (final_gas' = gas + "+str(gas)+")")
                        if "pos" in p[1]:
                            self.f.write(" & (positive' = true) ")
                        else:
                            self.f.write(" & (negative' = true) ")
                    self.f.write("; \n")

    def write_to_prism(self, write_attributes = False, steps_max = 50, max_gas = 1000, min_gas = -1000, write_parameterized = False, write_extended_parameterized=False):
        """Main function to write all information to given file in __init__.
        """
        file_path = self.f.name
        self.f.close()
        self.f = open(file_path, "w+")
        self.f.write('smg \n')
    
        # write global variables
        
        self.f.write('global gas : [-10000..10000] init 0; \n')
        self.f.write('global state : [0..100000] init 0; \n')
        
        self.f.write('global negative : bool; \n')
        self.f.write('global positive : bool; \n')
        self.f.write('global final_gas : [-1000..1000] init -10000; \n')
        self.f.write('global steps : [0..1000] init 0; \n')
        self.f.write('global min_gas : [0..1000] init 0; \n')

        if write_parameterized:
            self.f.write('const double envprob; \n')

        if write_extended_parameterized:
            self.f.write('const int m0; \n')
            self.f.write('const int m1; \n')
            self.f.write('const int m2; \n')

        edges_user = [t for t in self.g.edges if 'controllable' in self.g.edges[t] and not self.g.edges[t]['controllable'] 
                    and not self.g.edges[t]['action'] == "env"]
        edges_company = [t for t in self.g.edges if 'controllable' in self.g.edges[t] and self.g.edges[t]['controllable'] 
                    and not self.g.edges[t]['action'] == "env"]
        edges_env = [t for t in self.g.edges if self.g.edges[t]['action'] == "env"]
        
        reward_dicts = {'steps' : {}, 'gas':{}}
                
        self.f.write('module userModule \n')
        outgoing_edges = self.get_outgoing_edges(edges_user)
        self.write_edges(outgoing_edges, reward_dicts, write_attributes=write_attributes, steps_max=steps_max, max_gas=max_gas, min_gas=min_gas)
        self.f.write('endmodule \n')
        
        self.f.write('player userPlayer userModule ')
        for state in outgoing_edges:
            for action in outgoing_edges[state]:
                self.f.write(', [' + str(state) + 'SEP' + str(action) + ']')
        self.f.write('endplayer \n')

        # write provider module
        self.f.write('module providerModule \n')
        outgoing_edges = self.get_outgoing_edges(edges_company)
        self.write_edges(outgoing_edges, reward_dicts, write_attributes=write_attributes, steps_max=steps_max, max_gas=max_gas, min_gas=min_gas)
        self.f.write('endmodule \n')
        
        self.f.write('player providerPlayer providerModule ')
        for state in outgoing_edges:
            for action in outgoing_edges[state]:
                self.f.write(', [' + str(state) + 'SEP' + str(action) + ']')
        self.f.write('endplayer \n')
            
        # write environment module
        self.f.write('module controll \n')
        outgoing_edges = self.get_outgoing_edges(edges_env)
        self.write_edges(outgoing_edges, {'steps':{}, 'gas':{}}, write_parameterized = write_parameterized)

        # self-loops in terminal states 
        for s in self.terminal:
            self.f.write("[] state="+ self.states[s] + " -> (state' = "+ str(self.states[s])+")")
            if "pos" in s:
                self.f.write(" & (positive' = true) ")
            if "neg" in s:
                self.f.write(" & (negative' = true) ")
            self.f.write("; \n")
        
        self.f.write('endmodule \n')

        # write rewards
        self.f.write('rewards "steps" \n')
        for s in reward_dicts['steps']: 
            self.f.write("state=" + s + " : " + str(reward_dicts['steps'][s]) +"; \n")
        self.f.write('endrewards \n')

        self.f.write('rewards "gas_pos" \n')
        for s in reward_dicts['steps']: 
            self.f.write("state=" + s + " : " + str(max(0, reward_dicts['gas'][s])) +"; \n")
        self.f.write('endrewards \n')

        self.f.write('rewards "gas_neg" \n')
        for s in reward_dicts['steps']: 
            self.f.write("state=" + s + " : " + str(-1*min(0, reward_dicts['gas'][s])) +"; \n")
        self.f.write('endrewards \n')

        self.f.flush()

class PrismQuery(PrismPrinter):
    def __init__(self, g, path, name, PRISM_PATH : str):
        super().__init__(g, path, name)
        self.PRISM_PATH = PRISM_PATH

    def query(self, query_path : str, write_attributes = False, steps_max = 50, max_gas = 1000, min_gas = -1000, write_parameterized = False, envprob = 0):
        """_summary_

        Args:
            query_path (str): _description_
            write_attributes (bool, optional): Writing attributes as gas and steps - requires suitable maximum values. Defaults to False.
            steps_max (int, optional): Maximum number of steps to consider. Defaults to 50.
            write_parameterized (bool, optional): Write every transition parameterized, if not used, implies envprob = 0 - as observed in the logs. Defaults to False.
            envprob (int, optional): If set to 1, uses probabilities as observed, otherwise scales towards . Defaults to 1.

        Returns:
            _type_: _description_
        """
        self.write_to_prism(write_attributes=write_attributes, steps_max=steps_max, max_gas=max_gas, min_gas=min_gas, write_parameterized=write_parameterized)
        intermediate_output_file = open(self.path + "/intermediate_output", "w")
        intermediate_error_file = open(self.path + "/error_output", "w")
        call = [self.PRISM_PATH, self.f.name, query_path, "-verbose"]
        if write_parameterized:
            call.append('-const')
            call.append('envprob='+str(envprob))
            call.append('-javamaxmem')
            call.append('8g')
        subprocess.run(call, stdout = intermediate_output_file, stderr=intermediate_error_file)
        intermediate_output_file.close()
        intermediate_error_file.close()

        #assert no error occurred
        if stat(self.path + "/error_output").st_size != 0:
            print("errors occured:")
            with open(self.path + "/error_output") as file:
                lines = [line.rstrip() for line in file]
            print(lines)
            assert(False)

        # process results
        start_index = [True if '(non-zero only) for all states:' in line else False for line in open(self.path + "/intermediate_output")].index(True)+1
        df = pd.read_csv(self.path + "/intermediate_output", skiprows=start_index, header=None, skipfooter=6, engine ="python")
        df[0] = df[0].str.split("(").str[1]
        h = df[6].str.split("\)=")
        df[6] = h.str[0]
        df[7] = h.str[1]

        # convert types
        df = df.rename(columns = {0: "gas", 1: "state", 2:"negative", 3:"positive", 4:"finalGas", 5:"steps", 6:"min_gas", 7:"result"})
        df["gas"] = pd.to_numeric(df["gas"])
        df["state"] = pd.to_numeric(df["state"])
        df["finalGas"] = pd.to_numeric(df["finalGas"])
        df["steps"] = pd.to_numeric(df["steps"])
        df["min_gas"] = pd.to_numeric(df["min_gas"])
        df["result"] = pd.to_numeric(df["result"])

        # check if all unrolled states have same result
        results_file = {}
        int_states = {}
        for s in self.states:
            int_states[s] = int(self.states[s])
        for s in int_states:
            state_entry = df[df["state"] == int_states[s]]
            if state_entry.empty:
                assert(s in self.states)
                results_file[s] = 0
                continue
            results_file[s] = list(state_entry["result"])[0]
            if write_attributes:
                # account for rounding errors in 5th place
                assert len(set([round(r,5) for r in state_entry["result"]])) == 1, set([round(r,6) for r in state_entry["result"]])
            else:
                assert(len(state_entry["result"]) == 1)

        # assert that all nodes in graph have an associated result
        for s in self.g:
            assert(s in results_file)

        return results_file
    
    def get_strategy(self, query_path : str, debug = True):
        """Returned strategy dict mappsy states -> actions, includes environment and user actions.

        Args:
            query_path (str): Path to query to construct strategy for.

        Returns:
            dict: State to aciton mapping
        """
        self.write_to_prism()
        strategy_file = self.path + "strategy_file"
        intermediate_error_file = open(self.path + "/error_output", "w")
        subprocess.run([self.PRISM_PATH, self.f.name, query_path, "--exportstrat", strategy_file+".txt"], stdout=subprocess.DEVNULL, stderr=intermediate_error_file)
        subprocess.run([self.PRISM_PATH, self.f.name, query_path, "--exportstrat", strategy_file+".dot:type=dot,mode=restrict"], stdout=subprocess.DEVNULL, stderr=intermediate_error_file)
        intermediate_error_file.close()

        # use graph to filter unreachable states out
        g = nx.DiGraph(nx.nx_pydot.read_dot(strategy_file+".dot"))
        induced_states = []
        for s in g.nodes:
            if "label" in g.nodes[s]:
                if "(" in g.nodes[s]['label']:
                    induced_states.append(g.nodes[s]['label'].split(',')[1])
        # assert that no error occurred
        if stat(self.path + "/error_output").st_size != 0:
            print("errors occured:")
            with open(self.path + "/error_output") as file:
                lines = [line.rstrip() for line in file]
            print(lines)
            assert(False)
        strat_df = pd.read_csv(strategy_file+".txt", sep = "=", header = None)
        
        strat_df = strat_df.dropna() # exclude not reached states
        strat_df[0] = strat_df[0].str.split(",").str[1]
        strategy = {}
        strat_states = list(strat_df[0])
        for s in strat_states:
            # is now int, not str as in self.states
            original_state = [o_s for o_s in self.states if self.states[o_s] == s]
            if s not in induced_states:
                if debug:
                    print("skipped", original_state)
                continue 
            assert(len(original_state) == 1)
            original_state = original_state[0]
            actions = strat_df[strat_df[0] == s][1].values
            assert(len(actions)) == 1
            strategy[original_state] = actions[0].split("SEP")[1]
        return strategy