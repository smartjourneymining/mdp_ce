import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src import game_construction

# Constructs feature-dataframe from system.
# Colums correspond to events, cell is one if transitions occures, otherwise 0 (one-hot encoding of trace).
def to_df(log, system, history, abstraction):
    columns = set()
    for trace in log:
        columns.update(set([t['concept:name'] for t in trace]))
    #columns = list(set([system.edges[e]['action'] for e in system.edges()]))#list(system.edges())
    assert("start" in columns)
    columns = list(columns)
    trace_df = pd.DataFrame(columns = columns)#list(system.edges()).append('y'))
    for trace in log:
        trace_edges = [x['concept:name'] for x in trace]
        data = {"start":1}
        assert(trace[0]['concept:name']=="start")
        for i in range(1,len(trace_edges)):
            t = abstraction(trace[max(0,i-history+1):i+1])
            current = trace_edges[i]
            if t not in list(system.nodes()):
                continue
            if current not in data:
                data[current] = 1
        # ensure that every feature in contained
        for d in data:
            if d not in trace_df.columns:
                print(d)
            assert(d in trace_df.columns)
        # append to dataframe
        trace_df = pd.concat([trace_df, pd.DataFrame(data, index = [0])], ignore_index=True)
    trace_df = trace_df.fillna(0)
    assert(len(trace_df.index) == len(log))
    assert(len(trace_df.columns) == len(columns))
    return trace_df

# Compare correlations before and after decision boundary reduction.
# Dataframes are constructed from underlying systems and their log.
def analyse_correlations(log_application, original_system, reduced_system, name, history, abstraction, save_elements = True):
    # analyse used process model with mutli-set refinement and history 5
    trace_df = to_df(log_application, original_system, history, abstraction)
    trace_df_reduced = to_df(log_application, reduced_system, history, abstraction)
    
    for c in trace_df.columns:
        if((trace_df[c] == 0).all()):
            print("Original frame does not contain feature:", c)
            assert(False) # all features should be by definition contained
    for c in trace_df_reduced.columns:
        if((trace_df_reduced[c] == 0).all()):
            print("Reduced frame does not contain feature:", c)

    x = [x for x in trace_df.corr().unstack()]
    y = [x for x in trace_df_reduced.corr().unstack()]
    c, bins, bars = plt.hist([x, y], 20, label=['Original', 'Reduced'])
    counts = c[0]
    counts1 = c[1]

    print("Remaining correlations in % after reduction", [counts1[i]/counts[i] if counts[i] != 0 else np.inf if counts1[i] != 0 else np.nan for i in range(len(counts))])

    plt.xlabel("Pearson Correlation", fontsize = 18)
    plt.ylabel("Occurrences", fontsize = 18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.subplots_adjust(bottom=0.12)
    plt.legend(prop={'size':14})
    if save_elements:
        plt.savefig(name, dpi = 300)
    plt.clf()

    print(counts, counts1)
    for i in range(len(counts)):
        print("bin", bins[i], "--", counts1[i]/counts[i] if counts[i] != 0 else "inf" if counts1[i] != 0 else "nan")
