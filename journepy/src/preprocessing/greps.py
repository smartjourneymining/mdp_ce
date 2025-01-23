import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(
path=path, # path to the root directory
project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
dotenv=True, # load environment variables from .env if exists in root directory
pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
cwd=True, # change current working directory to the root directory (helps with filepaths)
)
import pandas as pd
import copy
from pm4py.objects.log.util import dataframe_utils
from collections import Counter
import argparse

from journepy.src.preprocessing.preprocessing_utils import *

# # Preprocessing
# The data is loaded and preprocessed.
# Names of events are parsed and unused events are removed.

def process_touchpoint_names(df):
    """Removes unused events and parses event names

    Args:
        df (pandas.DataFrame): Parsed datagrame of Greps csv file

    Returns:
        pandas.DataFrame: Dataframe with parsed and removed names
    """
    print("Found types", set(df['Type']))
    running_df = df[df['Type'].isin(['state', 'feedback','subject', 'task', 'resultsShared'])].copy()
    #prelims = df[df['concept:name'].str.contains("Preliminary results updated for overall score")] # include prelim results - doesnt make a lot of sense...
    #running_df = running_df.append(prelims)
    running_df['concept:name'] = running_df['concept:name'].str.replace('\n', "") # rows might contain line-breaks; for different A, B, C tasks
    running_df['concept:name'] = running_df['concept:name'].str.replace(r"loggedIn.*","loggedIn", regex=True) # removes feedback attributes
    running_df['concept:name'] = running_df['concept:name'].str.replace(r"\[.*\]","", regex=True) # removes feedback attributes
    running_df['concept:name'] = running_df['concept:name'].str.replace(",", "")
    running_df['concept:name'] = running_df['concept:name'].str.split('version').str[0]
    
    running_df['concept:name'] = running_df['concept:name'].str.split('Time used').str[0]

    running_df['concept:name'] = running_df['concept:name'].str.split('taskDownloaded').str[0]

    running_df['concept:name'] = running_df['concept:name'].str.split('testCompleted').str[0]

    running_df['concept:name'] = running_df['concept:name'].str.split('itemCompleted').str[0]

    running_df['concept:name'] = running_df['concept:name'].str.strip()

    m = running_df['concept:name'].str.contains("survey:")
    running_df.loc[m, 'concept:name'] = running_df['concept:name'][m].str.split('survey: ').str[1]
    
    m = running_df['Type'] == 'state'
    running_df.loc[m, 'concept:name'] = running_df['concept:name'][m].values
    running_df.loc[~m, 'concept:name'] = running_df['concept:name'][~m].values
    return running_df

def get_filtered_df(path, lower_bound = 15, upper_bound = 60):
    """Calls process_touchpoint_names and filters the dataset to only contain traces with length in [lower_bound, upper_bound].
    Prints information on successful and unsuccessful develpers.

    Args:
        path (Str): Path to greps data.csv file
        lower_bound (int, optional): Lower bound on length. Defaults to 15.
        upper_bound (int, optional): Upper bound on length. Defaults to 60.

    Returns:
        pandas.DataFrame: Processed dataframe
    """
    df = pd.read_csv(path, sep = None, engine='python') # infers delimiter automatically
    log_csv = dataframe_utils.convert_timestamp_columns_in_df(df)
    log_csv = log_csv.sort_values('Timestamp', kind = "stable")

    #rename columns to process mining notation
    log_csv.rename(columns={'Developer ID': 'case:concept:name', 'Message' : 'concept:name', 'Timestamp':'time:timestamp'}, inplace=True)
    log_csv['time:timestamp'] = pd.to_datetime(log_csv['time:timestamp'], unit='s')
    log_csv['case:concept:name'] = log_csv['case:concept:name'].astype(str)

    developers_finished_ids = set(log_csv[log_csv['concept:name'] == "finished"]['case:concept:name'].sort_values(kind = "stable").values)
    # process touchpoint names
    developers_df = process_touchpoint_names(log_csv)

    #filter used logs
    developers_df = developers_df.groupby(['case:concept:name']).filter(lambda x: len(x) >= lower_bound and len(x) <= upper_bound)

    unsuccesfull = developers_df[~developers_df['case:concept:name'].isin(list(developers_finished_ids))]['case:concept:name'].value_counts().to_list()
    succesfull = developers_df[developers_df['case:concept:name'].isin(list(developers_finished_ids))]['case:concept:name'].value_counts().to_list()
    print("Includes #unsuccesfull:", len(unsuccesfull),"and #succesfull", len(succesfull))
    developers_df = developers_df.sort_values(by=['time:timestamp'], kind = "stable")

    return developers_df

def build_log(df, max_duration = 180):
    """A log consists of traces, where each trace contains all elements from one developer, constructs list in xes format.
    Filtering performed:
    - Sorst logging in into different phases
    - Add tasks numbers
    - Add feedback numbers

    Args:
        df (pandas.DataFrame): DataFrame returned by get_filtered_df
        max_duration (int, optional): Max duration in days for journey to be considered. Defaults to 180.

    Returns:
        List[List[Dict]]]: Processed log in xes - list form
    """    """"""
    df_as_log = pm4py.convert_to_event_log(df)
    log = []
    running_df = df.copy()
    running_df = running_df.sort_values(by=['time:timestamp'], kind = "stable")

    for trace in df_as_log:
        current_trace = [i for i in trace]
        for i in current_trace:
            i['case:concept:name'] = trace.attributes['concept:name']
        if ((trace[-1]['time:timestamp']-trace[0]['time:timestamp']).days > max_duration):
            print("omitted developer", trace.attributes['concept:name'], "due to length")
            continue
        helper_element = copy.deepcopy(current_trace[0])
        helper_element['case:concept:name'] = trace.attributes['concept:name']
        helper_element["concept:name"] = "start"
        current_trace.insert(0, helper_element) #insert start event
        if contains(trace, "finished"):
                helper_element = copy.deepcopy(current_trace[-1])
                helper_element['case:concept:name'] = trace.attributes['concept:name']
                helper_element["concept:name"] = "positive"
                current_trace.append(helper_element)
        else:
            helper_element = copy.deepcopy(current_trace[-1])
            helper_element['case:concept:name'] = trace.attributes['concept:name']
            helper_element["concept:name"] = "negative"
            current_trace.append(helper_element)
        log.append(current_trace)


    # alter "logged in: Web page" to determine phase of journey:
    # Phase (1) sign up, (2) solve all programming tasks, and  (3) review and share the skill report with the customer.
    for t in log:
        indices = [i for i, x in enumerate(t) if x["concept:name"] == "Logged in: Web page"]
        for i in indices:
            t[i]["concept:name"] = "Logged in: Web page - Sign up"
            #result = np.where(t["concept:name"] == "Task event:")
            indices_task = [i for i, x in enumerate(t) if x["concept:name"] == "Task event:"]
            if contains(t, "Task event:") and min(indices_task) < i:
                t[i]["concept:name"] = "Logged in: Web page - Task"
            #result = np.where(t == "waitingForResultApproval")
            indices_approval = [i for i, x in enumerate(t) if x["concept:name"] == "waitingForResultApproval"]
            if contains(t, "waitingForResultApproval") and min(indices_approval) < i:
                t[i]["concept:name"] = "Logged in: Web page - Approval"

    # add task number
    for t in log:
        indices_feedback = [i for i, x in enumerate(t) if x["concept:name"] == "Give feedback"]
        indices_task = [i for i, x in enumerate(t) if x["concept:name"] == "Task event:"]
        for i in indices_task:
            count_indices = [j for j in indices_feedback if j < i] # uses feedback to increase task counter after giving feedback
            t[i]["concept:name"] += " "+str(len(count_indices))
    # add feedback number        
    for t in log:
        indices_feedback = [i for i, x in enumerate(t) if x["concept:name"] == "Give feedback"]
        indices_task = [i for i, x in enumerate(t) if x["concept:name"] == "Task event:"]
        for i in indices_feedback:
            count_indices = [j for j in indices_feedback if j < i]
            t[i]["concept:name"] += " "+str(len(count_indices))
    """
    # add number to log in:
    for t in log:
        indices_feedback = [i for i, x in enumerate(t) if x["concept:name"] == "Give feedback"]
        indices_loggin = [i for i, x in enumerate(t) if x["concept:name"] == "Task event: loggedIn"]
        for i in indices_loggin:
            count_indices = [j for j in indices_feedback if j < i]
            t[i]["concept:name"] += " "+str(len(count_indices))
    """

    return log

def filter_doubles(log):
    """Removes double elements from log

    Args:
        log (List[List[Dict]]]): Log to filter

    Returns:
        _type_: _description_
    """
    filtered_log = []
    for index in range(len(log)):
        # remove sequential same elements
        trace = log[index]
        current_trace = [trace[0]]
        for pos in range(1,len(trace)):
            if trace[pos]['concept:name']==trace[pos-1]['concept:name']:
                continue
            current_trace.append(trace[pos])

        filtered_log.append(current_trace)
    return filtered_log

def ignore_loggin(log):
    remove_subject = ['Logged in: Web page', 'Task event: loggedIn']
    log = [list(filter(lambda e: not (any( rem in e['concept:name'] for rem in remove_subject)), current_trace)) for current_trace in log]
    return log

def preprocessed_log(greps_path, include_loggin = False, max_duration = 180):
    """Function to retrun preprocessed greps log

    Args:
        greps_path (Str): Path to greps file
        max_duration (int, optional): Max duration in days for journey to be considered. Defaults to 180.

    Returns:
        List[List[Dict]]]: Greps log in xes - list format
    """    """"""
    filtered_df_single = get_filtered_df(greps_path)

    log_single = build_log(filtered_df_single, max_duration=max_duration)

    if not include_loggin:
        log_single = ignore_loggin(log_single)
    log_single = filter_doubles(log_single)

    # variants in log
    print("Variants before removing trivial elements")
    event_log = [[t['concept:name'] for t in trace] for trace in log_single]
    print(len(Counter(str(e) for e in event_log).keys()))

    return log_single

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'greps',
                    description = "Preprocesses the Greps event log.",)
    parser.add_argument('input', help = "Input Greps .csv file.")
    parser.add_argument('output', help = "Output path for processed log.")
    parser.add_argument('-m', '--max_duration', help = "Maximum duration of a single journey.", default = 180) 
    args = parser.parse_args()

    # generate preprocessed, discretized list of traces
    log = preprocessed_log(args.input, args.max_duration)
    # write as xes file - is not processed further
    export(log, args.output)