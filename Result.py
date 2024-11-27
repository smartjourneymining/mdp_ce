import pandas as pd

class Result:
    time = -1
    value = -1
    target_prob = -1
    timeout = -1
    gap = -1
    strategy = {}
    status = -1 # https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html
    
    def __init__(self, time, value, target_prob, strategy, timeout, gap, status = -1):
        self.time = time
        self.value = value
        self.target_prob = target_prob
        self.strategy = strategy
        self.timeout = timeout
        self.gap = gap
        self.status = status
        
    def df(self):
        d = {'time' : self.time, 'value' : self.value, 'target_prob' : self.target_prob, 'timeout' : self.timeout, 'gap' : self.gap, 'status' : self.status}
        return pd.DataFrame([d])