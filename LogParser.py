import abc
from aalpy.learning_algs import run_Alergia
from aalpy.utils import save_automaton_to_file
from aalpy.learning_algs import run_JAlergia
from IPython.display import Image

from journepy.src.preprocessing.greps import preprocessed_log as preprocessed_log_greps
from journepy.src.preprocessing.bpic12 import preprocessed_log as preprocessed_log_bpic_2012
from journepy.src.preprocessing.bpic17 import preprocessed_log as preprocessed_log_bpic_2017
from journepy.src.preprocessing.spotify import preprocessed_log as preprocessed_log_spotify
from journepy.src.alergia_utils import convert_utils

import json
import random 
import os

class LogParser:
    log_path = ""
    activities_path = ""
    data_environment = None 
    
    def __init__(self, log_path, activities_path):
        self.log_path = log_path
        self.activities_path = activities_path
        
    def automata_learning(self):       
        model = run_Alergia(self.data_environment, automaton_type='mdp', eps=0.1, print_info=True)
        save_automaton_to_file(model, "out/model.png", file_type="png")
        return model
    
    @abc.abstractmethod
    def preprocessing(self):
        return 
    
    @abc.abstractmethod
    def event_to_actor_file(self, event):
        return 

    def event_to_action_name(self, event):
        return event
    
    def event_to_activity(self, actors, action):
        # # build action mapping: assigns each event to an actor
        # actions_to_activities = {}
        # for a in actors:
        #     if actors[a] == "company":
        #         # if a in ['vpcAssignInstance', 'Give feedback 0', 'Results automatically shared', 'waitingForActivityReport']: # todo: might be quite realistic?
        #         #     actions_to_activities[a] = "company"
        #         # else:  
        #         #     actions_to_activities[a] = a
        #         actions_to_activities[a] = 'company'
        #     else:
        #         # if a == "negative":
        #         #     actions_to_activities[a] = "user"
        #         # elif "Give feedback" in a or "Task event" in a:
        #         #     actions_to_activities[a] = a
        #         # else:
        #         #     actions_to_activities[a] = "user"
        #         actions_to_activities[a] = a
        # return actions_to_activities[action]    
        
        # if action controlled by company: take 'company' action to learn distribution, user is deterministic
        if actors[self.event_to_actor_file(action)] == 'company':
            return 'company'
        else:
            return self.event_to_action_name(action)
            
        
    def build_benchmark(self):
        # load actor mapping: maps events to an actor (service provider or user)
        with open(self.activities_path) as f:
            data = f.read()
        actors = json.loads(data)

        filtered_log_activities = self.preprocessing()
        
        # TODO why mapped to actors naming? - should be adjusted to event_to_activity mapping above
        data = [[(self.event_to_activity(actors, t[i]), t[i]) for i in range(1, len(t))] for t in filtered_log_activities]
        for d in data:
            d.insert(0, 'start')
            
        # quantify environment - distribution of players after for events is learned
        data_environment = []
        for trace in data:
            current = [trace[0]]
            for i in range(1, len(trace)):
                e = trace[i]
                previous_state = "start" if i == 1 else trace[i-1][1]
                
                # encode decision in one step
                current.append(('env', actors[self.event_to_actor_file(e[1])] + previous_state))
                current.append(e)
            data_environment.append(current)
            
        self.data_environment = data_environment
        
        model = self.automata_learning()
        model = convert_utils.mdp_to_nx(model, actors)
    
        return model
    
class GrepsParser(LogParser):
    def preprocessing(self):
        filtered_log = preprocessed_log_greps(self.log_path, include_loggin=False) # also discards task-event log-in
        filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log]
        return filtered_log_activities
    
    def event_to_actor_file(self, event):
        return event
    
    def automata_learning(self):
        model = run_Alergia(self.data_environment, automaton_type='mdp', eps=0.1, print_info=True)
        save_automaton_to_file(model, "out/model.png", file_type="png")
        return model
    
class BPIC12Parser(LogParser):
    def preprocessing(self):
        filtered_log = preprocessed_log_bpic_2012(self.log_path)
        filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log]
        return filtered_log_activities
    
    def event_to_action_name(self, event):
        return self.event_to_actor_file(event)
    
    def event_to_actor_file(self, event):
        return event.split('#')[0]
    
    def automata_learning(self):
        model = run_Alergia(self.data_environment, automaton_type='mdp', eps=0.9, print_info=True)
        save_automaton_to_file(model, "out/model.png", file_type="png")
        return model
    
class BPIC17Parser(LogParser):
    
    def event_to_action_name(self, event):
        return self.event_to_actor_file(event)
    
    def event_to_actor_file(self, event):
        if 'W_Call after offers SUPER LONG' in event or 'W_Call incomplete files SUPER LONG' in event:
            return ' '.join(event.split(' ')[:-2])
        if "O_Create Offer" in event or 'W_Call after offers' in event or 'W_Call incomplete files' in event:
            return ' '.join(event.split(' ')[:-1])
        return event
    
    def automata_learning(self):
        model = run_Alergia(self.data_environment, automaton_type='mdp', eps=0.9, print_info=True)
        save_automaton_to_file(model, "out/model.png", file_type="png")
        return model
    
class BPIC17BothParser(BPIC17Parser):
    def preprocessing(self):
        filtered_log_before, filtered_log_after = preprocessed_log_bpic_2017(self.log_path) # uses common preprocessing
        filtered_log_before.extend(filtered_log_after)
        filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log_before]
        return filtered_log_activities
    
class BPIC17BeforeParser(BPIC17Parser):
    def preprocessing(self):
        filtered_log_before, filtered_log_after = preprocessed_log_bpic_2017(self.log_path) # uses common preprocessing
        filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log_before]
        return filtered_log_activities
    
class BPIC17AfterParser(BPIC17Parser):
    def preprocessing(self):
        filtered_log_before, filtered_log_after = preprocessed_log_bpic_2017(self.log_path) # uses common preprocessing
        filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log_after]
        return filtered_log_activities
    
class SpotifyParser(LogParser):
    def preprocessing(self):
        return preprocessed_log_spotify(self.log_path)
    
    number_samples = 0
    def __init__(self, log_path, activities_path, number_samples):
        super().__init__(log_path, activities_path)
        self.number_samples = number_samples
    
    def event_to_actor_file(self, event):
        if 'played' in event:
            return 'played'
        if 'song' in event:
            return 'song'
        if 'select_context' in event:
            return 'select_context'
        if 'startprofile' in event:
            return 'startprofile'
        if 'pause' in event:
            return 'pause'
        return event
    
    # def event_to_activity(self, actors, action):
    #     if 'song' in action:
    #         return action
    #     return 'company'
    
    def automata_learning(self):
        storage_file = f'out/spotify_{random.randint(0, 10000000000)}.txt'
        while os.path.isfile(storage_file):
            storage_file = f'out/spotify_{random.randint(0, 10000000000)}.txt'
        data = random.sample(self.data_environment, min(len(self.data_environment), self.number_samples))
        assert len(set([d[0] for d in self.data_environment])) == 1, f'Found start symbols {set([d[0] for d in self.data_environment])}'
        with open(storage_file, 'w') as f:
            for line in data:                
                f.write(line[0] + ',' + ','.join([e[0] + ',' + e[1] for e in line[1:]]))
                f.write('\n')
        model = run_JAlergia(path_to_data_file=storage_file, automaton_type='mdp', eps=0.9,
                     path_to_jAlergia_jar='../jAlergia/alergia.jar', heap_memory='-Xmx8g')
        assert model, f'None model for file {storage_file}'
        os.remove(storage_file)
        # model = run_JAlergia(path_to_data_file=random.sample(self.data_environment, min(len(self.data_environment), self.number_samples)), automaton_type='mdp', eps=0.9,
        #              path_to_jAlergia_jar='../jAlergia/alergia.jar')
        # model = run_Alergia(random.sample(self.data_environment, min(len(self.data_environment), self.number_samples)), automaton_type='mdp', eps=0.9, print_info=True)
        # save_automaton_to_file(model, "out/model.png", file_type="png")
        
        return model