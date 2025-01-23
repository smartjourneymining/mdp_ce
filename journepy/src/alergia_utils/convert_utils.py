from aalpy.automata import Mdp
import networkx as nx

def mdp_to_nx(model : Mdp, actors, debug = False):
    g = nx.DiGraph()
    for s in model.states:
        source = f"{s.state_id}: {s.output}"
        if debug:
            print("source:", source)
        # s.transitions: dict mapping available actions to lists of (states, prob)
        for action in s.transitions:
            if debug:
                print("for action:", action)
            for poss_targets in s.transitions[action]: # list of (state, prob)
                # poss_targets[0] - state
                # poss_targets[1] - prob
                target_state = f"{poss_targets[0].state_id}: {poss_targets[0].output}"
                if debug:
                    print("reaching state", target_state)
                    print("with prob:", poss_targets[1]) # reaching prob
                controllable = actors[action] == "company" if action in actors else action == "company"
                if debug:
                    print("is controllable", controllable)
                g.add_edge(source, target_state, action=action, prob_weight = poss_targets[1], controllable=controllable)
    return g