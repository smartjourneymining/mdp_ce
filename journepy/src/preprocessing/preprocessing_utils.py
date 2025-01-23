from collections import Counter
import pandas as pd
import pm4py

# 
def variants(log):
    """Compute number of variants in log (list of traces)

    Args:
        log (List[List[Dict]]]): Input log

    Returns:
        int: Number of variant while ignoring all information besides concept:name
    """
    element_list = []
    for trace in log:
        new_trace = [k["concept:name"] for k in trace]
        element_list.append(new_trace)
    return len(Counter(str(e) for e in element_list).keys())


def export(log, path):
    """Export a list of traces (each element is dict) for easier further analysis
       Assumes that every trace start with "start" and ends with "negative" or "positive"

    Args:
        log (List[List[Dict]]]): Log in xes - list format.
        path (Str): Path to write log to
    """
    # to use the pm4py write_xes function, an pm4py event log object is needed.
    # for this is the list event log via the pandas data frame to a pm4py event log converted.
    log_df = pd.DataFrame()
    for trace in log:
        assert(trace[0]["concept:name"] =="start")
        assert(trace[-1]["concept:name"] == "negative" or trace[-1]["concept:name"]== "positive")
        log_df = pd.concat([log_df, pd.DataFrame(trace)], ignore_index=True)

    log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'], utc = True)
    pm4py.write_xes(log_df, path) 

def contains(trace, element):
    """Helpfer function to check if element is contained in trace

    Args:
        trace (List[Dict]): Trace in xes file format
        element (Str): tElement searched for

    Returns:
        _type_: _description_
    """
    for event in trace:
        if event['concept:name']==element:
            return True
    return False