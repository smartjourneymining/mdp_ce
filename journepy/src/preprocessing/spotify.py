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
#import preprocessing_utils
from journepy.src.preprocessing.preprocessing_utils import *



def assert_implications(df):
    # Assert implications:
    # not_skipped -> not (skip_1 or skip_2 or skip_3)
    # skip_i -> not not_skipped
    for i in range(df.shape[0]):
        line = df.iloc[i]
        if line['not_skipped']:
            # if not skipped, all skip_i are false
            assert(not (line['skip_1'] or line['skip_2'] or line['skip_3']))
        else:
            if line['skip_3']:
                #print(line)
                #assert(not line['skip_2'] and not line['skip_1'])
                assert(not line['not_skipped'])
            if line['skip_2']:
                #assert(not line['skip_1'])
                assert(not line['not_skipped'])
            if line['skip_1']:
                assert(not line['not_skipped'])

        if line['no_pause_before_play']:
            assert(not line['short_pause_before_play'] and not line['long_pause_before_play'])
        if not line['no_pause_before_play']:
            if line['session_position']!= 1:
                assert(line['short_pause_before_play'] or line['long_pause_before_play'])


def get_profile(session_df : pd.DataFrame, prefix : str, feature_names : list):
    avg_df = session_df.mean(numeric_only=True)
    features = []
    for feature_name in feature_names:
        features.append(str(int(10*avg_df[feature_name])))
    profile = "_".join(features)
    return [prefix+'profile_'+profile]


def build_journeys(df, feature_names : list, min_track_plays, controllable_context):
    log = []
    song_durations = {}
    grouped_df = df.groupby(by = ['session_id'])
    added = 0
    for session_id in set(df['session_id']):
        session_log = ['start_session']
        df_filtered = grouped_df.get_group(session_id)
        df_filtered = df_filtered.sort_values(['session_position'])
        df_filtered.reset_index(drop=True, inplace=True)
        df_dicts = df_filtered.to_dict(orient='index')
        # start profile
        session_log.extend(get_profile(df_filtered[df_filtered['session_position']<= 5], "start", feature_names))
        for pos in range(0, len(df_filtered['session_length'])):
            e = df_dicts[pos] # convert to single row structure
            assert(e['session_position'] == pos+1)
            # pauses
            if pos != 0: # if first pos, there was no break...
                if not e['no_pause_before_play']:
                    if e['long_pause_before_play']:
                        session_log.append("long_pause")
                    else:
                        assert(e['short_pause_before_play'])
                        session_log.append("short_pause")
            
            # always append context to determine if next song selection is controllable
            context = e['context_type']
            session_log.append('select_context_'+ context)
            
            decade = str(e['release_year']%100//10*10)
            duration = int(e['duration']//60)
            mode = str(e['mode'])
            features = []
            for feature_name in feature_names:
                features.append(str(int(10*e[feature_name])))
            prefix = 'played'
            if e['context_type'] in controllable_context:
                prefix = 'song'
            # state = "_".join([prefix, duration, decade, mode, acousticness, danceability, energy])
            
            state_features = [prefix, decade, mode]
            state_features.extend(features)
            state = "_".join(state_features)
            
            if state in song_durations:
                song_durations[state].add(duration)
            else:
                song_durations[state] = {duration}

            session_log.append(state)
            # skips - challenge targets for skip_2, decided to use
            # if pos < len(df_filtered['session_length']): # add only if not last song # TODO
            if not e['skip_2']:
                session_log.append("skipped")
            else:
                session_log.append("not_skipped")

        # we ignore forwarding and backwarding
        # end profile
        # dont append end - only insert in contexts
        if pos < 15: # TODO
            session_log.append('negative')
        else:
            session_log.append('positive')
        log.append(session_log)
    return log, song_durations

# filter profiles
def filter_log_profiles(log, start_profile_pair, target_profile_pair, feature_names, max_dist = 2):
    profiles = []
    assert(len(start_profile_pair) == len(feature_names))
    assert(len(target_profile_pair) == len(feature_names))
    filtered_log = []
    for t in log:
        assert 'profile_'  in t[1]
        profile = t[1].split('startprofile_')[1].split('_')
        profiles.append('_'.join(profile))
        assert len(start_profile_pair) == len(profile)
        if all([int(profile[i]) >= min(start_profile_pair[i], target_profile_pair[i]) - max_dist and 
                int(profile[i]) <= max(start_profile_pair[i], target_profile_pair[i]) + max_dist for i in range(len(profile))]):
            filtered_log.append(t)
    return filtered_log

def preprocessed_log(spotify_folder_path, start_profile = [5, 6, 4], target_profile = [0, 3, 9], feature_names = ["acousticness", "danceability", "energy"]):
    df_tracks = pd.read_csv(f'{spotify_folder_path}/track_features/tf_mini.csv')
    df = pd.read_csv(f'{spotify_folder_path}/training_set/log_mini.csv')
    df['no_pause_before_play'] = df['no_pause_before_play'].astype('bool')
    df['long_pause_before_play'] = df['long_pause_before_play'].astype('bool')
    df['short_pause_before_play'] = df['short_pause_before_play'].astype('bool')
    # years played in contained journeys
    merged = df.merge(df_tracks, left_on='track_id_clean', right_on='track_id')

    # enrich with other information
    df = merged
    
    # check assumptions
    print("Check implications")
    assert_implications(df)
    print("All passed")
    
    print("Build journeys")
    controllable_context = ['radio', 'editorial_playlist', 'charts', 'personalized_playlist']
    log, song_durations = build_journeys(df, feature_names, 19, controllable_context)
    # assume that the duration of a song is at least one time unit
    song_durations = dict((k, (max(min(v), 1), max(v))) for k, v in song_durations.items())

    #print("Filter profiles")
    #filtered_log = filter_log_profiles(log, start_profile, target_profile, feature_names, max_dist = 1)
    print("Len of filtered log", len(log))
     
    return log

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog = 'log_parser',
                    description = "Takes the Spotify event log as input and performs the preprocessing described in 'Nudging Strategies for User Journeys: Take a Path on the Wild Side' by Johnsen et al.",)
    parser.add_argument('input', help = "Input folder for Spotify event log") 
    parser.add_argument('output', help = "Output path for processed event logs") 
    parser.add_argument('start_profile', help = "Start profile to filter on", nargs='+', type=int, default = [5,6,4])
    parser.add_argument('target_profile', help = "Target profile to filter on", nargs='+', type=int, default = [0, 3, 9])
    parser.add_argument('feature_names', help = "Target profile to filter on", nargs='+', type=str, default = ["acousticness", "danceability", "energy"])
    args = parser.parse_args()

    discretized_list_log_before, discretized_list_log_after = preprocessed_log(args.input, args.start_profile, args.target_profile, args.feature_names)
    # write as xes file - is not processed further
    # export(discretized_list_log_before, args.output+"_before")