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
from pm4py.objects.log.importer.xes import importer as xes_importer
import copy
from journepy.src.preprocessing.preprocessing_utils import *
from sklearn import mixture
import numpy as np

"""
############################################################
In the preprocessing, we:
- transform the log into a list format
- discretise offer events
- remove trivial events
- remove incomplete and declines journeys
- compute durations of calls
- cluster call events 
#############################################################
"""

# investigate redundant elements
def verify_redundent_elements(log):
    """ Verifies known redundant elements:
    - A_PARTLYSUBMITTED is directly followed by A_SUBMITTED
    - O_DECLINED appears with A_DECLINED
    - If A_APPROVED three elements follow in arbitrary order: O_ACCEPTED A_ACTIVATED,A_REGISTERED; none of the before appears otherwise.
    - O_CREATED is directly followed by O_SENT

    Args:
        log (Pm4py.EventLog): Log to be checked
    """
    for trace in log:
        for i in range(len(trace)):
            if trace[i]['concept:name'] == "A_PARTLYSUBMITTED":
                assert(trace[i-1]['concept:name']== "A_SUBMITTED")

            """
            # cant confirm from Bautista et al.
            if trace[i]['concept:name'] == "O_CANCELLED":
                if not (trace[i+1]['concept:name'] == "A_CANCELLED" or trace[i-1]['concept:name'] == "A_CANCELLED"):
                    print(trace[i-2]['concept:name'])
                    print(trace[i-1]['concept:name'])
                    print(trace[i]['concept:name'])
                    print(trace[i+1]['concept:name'])
                    print(trace[i+2]['concept:name'])
                assert(trace[i+1]['concept:name']== "A_CANCELLED" or trace[i-1]['concept:name']== "A_CANCELLED")
            """

            if trace[i]['concept:name'] == "O_DECLINED":
                assert(trace[i+1]['concept:name']== "A_DECLINED" or trace[i-1]['concept:name']== "A_DECLINED")
            
            # in successfull traces always 4 events together; occur only in successfull traces
            if contains(trace, "A_APPROVED"):
                if trace[i]['concept:name'] == "A_APPROVED":
                    pos1 = 0
                    pos2 = 0
                    pos3 = 0

                    for j in range(len(trace)):
                        if trace[j]["concept:name"] == "O_ACCEPTED":
                            pos1 = j
                    for j in range(len(trace)):
                        if trace[j]["concept:name"] == "A_ACTIVATED":
                            pos2 = j
                    for j in range(len(trace)):
                        if trace[j]["concept:name"] == "A_REGISTERED":
                            pos3 = j

                    s = sorted([pos1, pos2, pos3, i])
                    assert(s[0] == s[1]-1 and s[1] == s[2]-1 and s[2] == s[3]-1 or not contains(trace, "O_ACCEPTED") and s[1] == s[2]-1 and s[2] == s[3]-1 ) 
                    # in three cases is "O_ACCEPTED" not contained, but then are the offers also pre-accepted and the other three elements are in group
            else:
                assert(not contains(trace, "A_APPROVED"))
                assert(not contains(trace, "O_ACCEPTED"))
                assert(not contains(trace, "A_ACTIVATED"))
                assert(not contains(trace, "A_REGISTERED"))
            
            if trace[i]['concept:name'] == "O_CREATED":
                assert(trace[i+1]['concept:name']== "O_SENT")

# investigate number of offers - has no event for "sent offer"
# discretise single 0_CREATED and 0_SENT events
def discretize_offers(log):
    """ Discretizes offers by counting the number of offers sent 

    Args:
        log (Pm4py.EventLog): Log to be discretized
    """
    for trace in log:
        offer_count = -1
        for e in trace:
            if "O_CREATED" in e["concept:name"]:
                offer_count += 1
                e["concept:name"] += str(offer_count)

            elif "O_SENT" in e["concept:name"] and "BACK" not in e["concept:name"]:
                e["concept:name"] += str(offer_count)

# Verify that counts align with paper and previous analysis
def confirm_counts(log):
    """ Comfirms that counts of final events allign with known distribution:
    - A_DECLINED: 7635
    - A_CANCELLED: 2807
    - A_APPROVED: 2246
    - unclass: 399

    Args:
        log (Pm4py.EventLog): Log to confirm outcome counts
    """
    counts = {"A_DECLINED":0, "A_CANCELLED":0, "A_APPROVED":0, "unclass": 0}
    for trace in log:
        found = False
        for k in list(counts.keys()):
            if contains(trace, k):
                assert(not found)
                found = True
                counts[k] += 1
        if not found:
            counts["unclass"] += 1
    assert(counts['A_DECLINED']==7635)
    assert(counts['A_CANCELLED']==2807)
    assert(counts['A_APPROVED']==2246)
    assert(counts['unclass']==399)

# Build list of dicts 
def log_to_list(log):
    """ Takes pm4py EventLog as input an returns a list of dicts.

    Args:
        log (Pm4py.EventLog): Log to map to list

    Returns:
        List[List[Dict]]]: List of dicts in same order as input and same dict contents
    """
    transformed_log = []
    for trace in log:
        current_trace = []
        for event in trace:
            # store case id in every event
            event['case:concept:name'] = trace.attributes['concept:name']
            current_trace.append(event)
        transformed_log.append(current_trace)
    return transformed_log

# A complete log is constructed by filtering out incomplete traces, adding a start state and final states (negative and positive)
# Original event log is untouched
def construct_complete_log(log):
    """ Inserts start and positive and negative final elements.

    Args:
        log (List[List[Dict]]]): Log to insert start and final elements

    Returns:
        List[List[Dict]]]: Same event logs as input but with start and final elements
    """
    transformed_log = []
    outcomes = ["A_CANCELLED", "A_APPROVED"] # "A_DECLINED",
    for trace in log:
        trace_copy = copy.deepcopy(trace)
        contained = False
        for o in outcomes:
            if contains(trace, o):
                contained = True
        if contained:
            trace_copy.insert(0, {"concept:name": "start", 'time:timestamp': trace[0]['time:timestamp'], 'case:concept:name': trace[0]['case:concept:name']}) #insert start event
            if contains(trace, "A_CANCELLED"):
                trace_copy.append({'concept:name': "negative", 'time:timestamp': trace[-1]['time:timestamp'], 'case:concept:name': trace[0]['case:concept:name']})
            if contains(trace, "A_APPROVED"):
                trace_copy.append({'concept:name': "positive", 'time:timestamp': trace[-1]['time:timestamp'], 'case:concept:name': trace[0]['case:concept:name']})
            assert(contains(trace_copy, "negative") or contains(trace_copy, "positive"))
            transformed_log.append(trace_copy)
    return transformed_log

def massage_log(log, min_speaking_time, day_difference, timeout_threshold):
    """Filter events from log and compute durations of single calls and differentiate between bank cancellation and user cancellation

    Args:
        log (List[List[Dict]]]): Log to massage
        min_speaking_time (int): min accumulated speaking time in seconds
        day_difference (int): day difference to differentiate between user cancellation and auto-company cancellation
        timeout_threshold (int): Difference to introduce timeout

    Returns:
        List[List[Dict]]]: massaged log
    """
    transformed_log = []
    for trace in log:
        new_trace  = []
        for i in range(len(trace)):
            if "W_Nabellen" in trace[i]['concept:name']: # omits SCHEDULE and COMPLETE call events
                if trace[i]['lifecycle:transition']=="START":
                    found = False
                    duration = 0
                    j = i
                    while not found:
                        j = j+1
                        if j >= len(trace):
                            for e in trace:
                                print(e["concept:name"], e['lifecycle:transition'])
                        assert(j < len(trace))
                        if trace[j]['concept:name'] == trace[i]['concept:name']:
                            assert(trace[j]['lifecycle:transition'] != "START")
                            if  trace[j]['lifecycle:transition']=="COMPLETE":
                                found = True
                                duration = (trace[j]['time:timestamp']-trace[i]['time:timestamp']).total_seconds()
                        
                    if duration > min_speaking_time:
                        if new_trace[-1]["concept:name"] == trace[i]['concept:name']: # merge call times together
                            new_trace[-1]["duration"] += duration
                        else:
                            new_element = copy.deepcopy(trace[i])
                            new_element["duration"] = duration
                            new_trace.append(new_element)
            if trace[i]['concept:name'] == "A_Cancelled": #differentiate between user_abort and timeout
                new_element = copy.deepcopy(trace[i])
                if (trace[i]['time:timestamp']-trace[i-1]['time:timestamp']).days >= day_difference or (trace[i]['time:timestamp']-trace[i-2]['time:timestamp']).days >= timeout_threshold :
                    new_element[-1]['concept:name'] = "TIMEOUT"
                    assert(False) # no timeouts detected before cancellation
                else:
                    new_element[-1]['concept:name'] += " CUSTOMER"
                new_trace.append(new_element)
            else:
                if "W_" in trace[i]['concept:name']:  # skip other workflow elements
                    continue
                if "O_SELECTED" in trace[i]['concept:name']:
                    continue
                if "O_SENT" in trace[i]['concept:name'] and "BACK" not in trace[i]['concept:name']: # skip sent event, but not send_back
                    continue
                elif "O_DECLINED" in trace[i]['concept:name'] or "A_PARTLYSUBMITTED" in trace[i]['concept:name']: # skip trivial elements
                    continue
                else:
                    new_trace.append(trace[i])

        transformed_log.append(new_trace)
    return transformed_log

def merge_successful(log):
    """Function to merge the 4 events present in successful logs: A_APPROVED, O_ACCEPTED, A_ACTIVATED and A_REGISTERED

    Args:
        log (List[List[Dict]]]): input log

    Returns:
        List[List[Dict]]]: log with merged success elements
    """
    transformed_log = []
    for trace in log:
        if contains(trace, "A_APPROVED"):
            modified_trace = copy.deepcopy(trace)

            for j in range(len(modified_trace)):
                if modified_trace[j]["concept:name"] == "O_ACCEPTED":
                    modified_trace.pop(j)
                    break

            for j in range(len(modified_trace)):
                if modified_trace[j]["concept:name"] == "A_ACTIVATED":
                    modified_trace.pop(j)
                    break
            
            for j in range(len(modified_trace)):
                if modified_trace[j]["concept:name"] == "A_REGISTERED":
                    modified_trace.pop(j)
                    break
            
            transformed_log.append(modified_trace)

        else:
            transformed_log.append(trace)
    return transformed_log


def get_bayesian_gaussian_mixture(components, times):
    """ For dicts of times, gaussion_mixture classifiers are constructed. 
    Returns a dict mapping events to classifier (for each e.g. type of call is a classifier learned)

    Args:
        components (int): Number of components in gaussion mixture
        times (Dict): Dicts containing durations(list of ints) for elements(keys)

    Returns:
        mixture.BayesianGaussianMixture: Mapping elements to classifiers (gaussion mixtures)
    """
    duration_classifier = {}
    for t in times:
        if len(times[t])==1:
            g = mixture.BayesianGaussianMixture(n_components=1,covariance_type='full')
            g.fit(np.array([[times[t]], [times[t]]]).reshape(-1, 1))
            duration_classifier[t] = g
            continue
        
        g = mixture.BayesianGaussianMixture(n_components=components,covariance_type='full', random_state=42)
        g.fit(np.array(times[t]).reshape(-1,1))
        duration_classifier[t] = g
    return duration_classifier

# Discretizes durations of calls by given classifiers
# 
def classify_log(log, classifier, cluster_components):
    """Discretizes durations of calls by given classifiers. Original log is unchanged and the discretized log is returned.

    Args:
        log (List[List[Dict]]]): Input log
        classifier (mixture.BayesianGaussianMixture): Classifiers constructed with `get_bayesian_gaussian_mixtur`
        cluster_components (int): Number of components to be casted into (only used in assertion)

    Returns:
        List[List[Dict]]]: Modified log with classified durations
    """
    log = copy.deepcopy(log)
    for trace in log:
        for pos in range(1,len(trace)):
            action = trace[pos]["concept:name"]
            if "W_Nabellen" not in action:
                continue
            assert("duration" in trace[pos])
            duration = trace[pos]["duration"]
            suffix = str(classifier[action].predict(np.array(duration).reshape(1,-1))[0])
            assert(int(suffix) in list(range(0, cluster_components)))
            trace[pos]["concept:name"] += "#"+suffix

    return log

def preprocessed_log(bpic12_path, min_speaking_time=60, day_difference=10, timeout_threshold=30, cluster_components=3):
    """Calls preprocessing and returns a discretized, reduced log

    Args:
        bpic12_path (str): Path to original bpic_12 file
        min_speaking_time (int, optional): Min speakingtime for single calls to be considered. Defaults to 60.
        day_difference (int, optional): Day difference for timeouts. Defaults to 10.
        timeout_threshold (int, optional): Day difference for timeouts. Defaults to 30.
        cluster_components (int, optional): Number of components in gaussian clustering. Defaults to 3.

    Returns:
        _type_: _description_
    """
    log = xes_importer.apply(bpic12_path)
    verify_redundent_elements(log)
    discretize_offers(log)

    # verify finding from Bautista etal. report
    confirm_counts(log)

    # transform log to list of traces
    list_log = log_to_list(log)

    # remove incomplete traces and add start and final states
    list_log = construct_complete_log(list_log)

    print("Variants before removing trivial elements", variants(list_log))
    assert(variants(list_log) == 3319)

    # add durations to talks, 
    list_log = massage_log(list_log, min_speaking_time, day_difference, timeout_threshold)

    # a successful trace consists always of 4 states
    list_log = merge_successful(list_log)

    print("Variants after removing trivial events", variants(list_log))
    assert(variants(list_log)==202)

    # build call durations - how long where calls?
    call_durations = {}
    for trace in list_log:
        for e in trace:
            if "W_Nabellen" in e['concept:name']:
                if e['concept:name'] not in call_durations:
                    call_durations[e['concept:name']] = []
                call_durations[e['concept:name']].append(e['duration'])

    classifier = get_bayesian_gaussian_mixture(cluster_components, call_durations)
    discretized_list_log = classify_log(list_log, classifier, cluster_components)

    return discretized_list_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'bpic12',
                    description = "Preprocesses the BPIC12 event log.",)
    parser.add_argument('input', help = "Input BPIC 2012 .xes file.")
    parser.add_argument('output', help = "Output path for processed log.")
    parser.add_argument('-mst', '--min_speaking_time', help = "Minimum duration of an aggregated call event to be considered (in sec.); default = 60", default = 60) 
    parser.add_argument('-d', '--day_difference', help = "Number of days until user is considered to abort journey; default = 10", default = 10) 
    parser.add_argument('-t', '--timeout-threshold', help = "Number of days until the cancellation is considered a timeout; default = 30", default = 30) 
    parser.add_argument('-cluster', '--cluster_components', help = "Number of components in Gaussian Mixture Clustering; default = 3", default = 3) 
    args = parser.parse_args()

    # generate preprocessed, discretized list of traces
    discretized_list_log = preprocessed_log(args.input, args.min_speaking_time, args.day_difference, args.timeout_threshold, args.cluster_components)
    # write as xes file - is not processed further
    export(discretized_list_log, args.output)