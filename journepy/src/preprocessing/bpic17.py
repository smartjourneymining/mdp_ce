import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(
path=path, # path to the root directory
project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
dotenv=True, # load environment variables from .env if exists in root directory
pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
cwd=True, # change current working directory to the root directory (helps with filepaths)
)
import argparse
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
#import preprocessing_utils
from journepy.src.preprocessing.preprocessing_utils import *


def filter_log(log):
    """ Filter function to remove outliers; all singleton traces

    Args:
        log (Pm4py.EventLog): Original log

    Returns:
        Pm4py.EventLog: Log with removed singleton variants
    """
    perc = 2/len(log)
    return pm4py.filter_variants_by_coverage_percentage(log, perc)

def construct_log(log, mst, d):
    """Construct event log by parsing into list of dicts and:
    - Discretizing to short, long or super longs talks
    - Skippeing W_ and "A_Created", "A_Complete", "A_Incomplete"
    - Differentiating A_Cancelled into TIMEOUT and CUSTOMER
    - Merging O_Create and O_Created
    - Merging double elements in ['W_Call incomplete files', 'W_Call after offers', 'W_Complete application', 'W_Validate application']
    - If A_Pending and O_Cancelled in same trace, remove all cancelled elements, one offer was accepted
    - End trace at first final_state event
    - Remove elements without final_state event

    Args:
        log (Pm4py.EventLog): Filtered event log
        mst (int): Minimum speaking time to be considered (in seconds)
        d (int): Threshold to differenciate between customer timeout and company timeout (in days)

    Returns:
        List[List[Dict]]]: Massaged event log
    """
    terminal_states = ['A_Cancelled COMPANY', 'A_Cancelled CUSTOMER', 'A_Pending', 'TIMEOUT']
    to_merge = ['W_Call incomplete files', 'W_Call after offers', 'W_Complete application', 'W_Validate application']
    log_activities = []
    for trace in log:
        current_trace = [trace[0]]
        current_trace[0]['case:concept:name'] = trace.attributes['concept:name']
        for i in range(1,len(trace)):
            pos = trace[i]
            pos['case:concept:name'] = trace.attributes['concept:name']
            if "W_Call" in trace[i]['concept:name']:
                # search for closing event
                if pos['lifecycle:transition'] in ["start", "resume"]:
                    for inner_index in range(i+1, len(trace)):
                        inner_pos = trace[inner_index]
                        if pos['concept:name'] == inner_pos['concept:name']:
                            if inner_pos['lifecycle:transition'] in ["suspend", "complete"]:                 
                                duration = (inner_pos['time:timestamp']-pos['time:timestamp']).total_seconds()
                                if duration > mst:
                                    if pos['concept:name'] in current_trace[-1]["concept:name"]:
                                        current_trace[-1]["duration"] += duration
                                    else:
                                        current_trace.append(pos)
                                        current_trace[-1]['duration'] = duration
                                    if current_trace[-1]["duration"] < 600:
                                        current_trace[-1]['concept:name'] = pos['concept:name']+" SHORT"
                                    elif current_trace[-1]["duration"] < 14400:
                                        current_trace[-1]['concept:name'] = pos['concept:name']+" LONG"
                                    else:
                                        current_trace[-1]['concept:name'] = pos['concept:name']+" SUPER LONG"
                            break
            if "W_" in trace[i]['concept:name']:
                continue # skip other workflow events
            if trace[i]['concept:name'] in ["A_Created", "A_Complete", "A_Incomplete"]:
                continue # skip trivial elements
            if trace[i]['concept:name'] == "A_Cancelled": #differentiate between user_abort and timeout
                current_trace.append(pos)
                if (trace[i]['time:timestamp']-trace[i-1]['time:timestamp']).days >= d:
                    current_trace[-1]['concept:name'] = "TIMEOUT"
                else:
                    current_trace[-1]['concept:name'] += " CUSTOMER"
                continue
            if "O_Created" == trace[i]['concept:name']:
                continue # merge create and created
            if trace[i]['concept:name'] in terminal_states:
                current_trace.append(pos)
            else:
                if trace[i]['concept:name'] in to_merge and trace[i]['concept:name'] == trace[i-1]['concept:name']:
                    continue
                else:
                    current_trace.append(pos)
        if "A_Pending" in [pos['concept:name'] for pos in current_trace]:
            if "O_Cancelled" in [pos['concept:name'] for pos in current_trace]:
                for pos1 in current_trace:
                    if 'O_Cancelled' in pos1['concept:name']:
                        current_trace.remove(pos1)
        intersection = [i for i in trace if i['concept:name'] in terminal_states]
        for state in terminal_states:
            indices = [i for i, x in enumerate(current_trace) if x['concept:name'] == state]
            if indices:
                current_trace = current_trace[:indices[0]+1]
        if intersection:
            log_activities.append(current_trace)
    
    return log_activities

def process_log(log):
    """Process log to iterate created offers and differentiate between positive and negative traces, operates inplace

    Args:
        log (List[List[Dict]]]): input log
    """
    MAX_INDEX = 100
    for trace in log:
        isPositive = False
        if contains(trace, 'A_Pending'):
            isPositive = True
        trace.insert(0,{'concept:name': 'start', 'case:concept:name': trace[0]['case:concept:name'], 'time:timestamp': trace[0]['time:timestamp']})
        if isPositive:
            trace.append({'concept:name': 'positive', 'case:concept:name': trace[0]['case:concept:name'], 'time:timestamp': trace[0]['time:timestamp']})
        else:
            trace.append({'concept:name': 'negative', 'case:concept:name': trace[0]['case:concept:name'], 'time:timestamp': trace[0]['time:timestamp']})
    
    to_extend = ["O_Create Offer"]
    for name in to_extend:
        for trace in log:
            indices = [i for i, x in enumerate(trace) if x['concept:name'] == name]
            for i in indices:
                count_indices = [j for j in indices if j < i]
                index = MAX_INDEX if len(count_indices) > MAX_INDEX else len(count_indices)
                trace[i]['concept:name'] += " "+str(index)


def preprocessed_log(bpic17_path, mst = 60, d = 20):
    """Returns two processed logs: splitted at concept drift

    Args:
        bpic17_path (Str): Path to original bpic17 event log
        mst (int, optional):  min_speaking_time in seconds. Defaults to 60.
        d (int, optional): day_timeout in days. Defaults to 20.

    Returns:
        _type_: _description_
    """
    # Load the log
    log = xes_importer.apply(bpic17_path)

    # split at concept drift
    log_before = pm4py.filter_time_range(log, "2011-03-09 00:00:00", "2016-06-30 23:59:59", mode='traces_contained')
    log_after = pm4py.filter_time_range(log, "2016-08-01 00:00:00", "2018-03-09 00:00:00", mode='traces_contained')

    # filter outliers
    filtered_log_before = filter_log(log_before)
    filtered_log_after = filter_log(log_after)

    # construct log - implicitly transforms into list of logs
    filtered_log_before = construct_log(filtered_log_before, mst, d)
    filtered_log_after = construct_log(filtered_log_after, mst, d)

    # append positive or negative outcome
    process_log(filtered_log_before)
    process_log(filtered_log_after)

    return filtered_log_before, filtered_log_after

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog = 'log_parser',
                    description = "Takes the BPIC'17 event log as input and performs the preprocessing described in 'Building User Journey Games from Multi-party Event Logs' by Kobialka et al. Outputs two event logs by default: before and after the concept drift in July.",)
    parser.add_argument('input', help = "Input file for BIPC'17 event log") 
    parser.add_argument('output', help = "Output path for processed event logs") 
    parser.add_argument('-mst', '--min_speaking_time', help = "Minimum duration of an aggregated call event to be considered (in sec.); default = 60", default = 60) 
    parser.add_argument('-d', '--day_timeout', help = "Number of days until the cancellation is considered a timeout; default = 20", default = 20) 
    args = parser.parse_args()

    discretized_list_log_before, discretized_list_log_after = preprocessed_log(args.input, args.min_speaking_time, args.day_timeout)
    # write as xes file - is not processed further
    export(discretized_list_log_before, args.output+"_before")
    export(discretized_list_log_after, args.output+"_after")