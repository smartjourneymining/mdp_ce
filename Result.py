class Result:
    time = -1
    value = -1
    target_prob = -1
    strategy = {}
    
    def __init__(self, time, value, target_prob, strategy):
        self.time = time
        self.value = value
        self.target_prob = target_prob
        self.strategy = strategy