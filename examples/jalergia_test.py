from aalpy.learning_algs import run_JAlergia
from aalpy.utils import visualize_automaton

from aalpy.learning_algs import run_Alergia
from aalpy.utils import save_automaton_to_file

data = []
with open('spotify100.txt') as f:
    data = f.read().splitlines()

data_parsed = []    
for d in data:
    split = d.split(',')
    split_1 = [(split[i], split[i+1]) for i in range(1, len(split), 2)]
    split_1.insert(0, split[0])
    data_parsed.append(split_1)
    
model = run_JAlergia(path_to_data_file='spotify100.txt', automaton_type='mdp', eps=0.9,
                     path_to_jAlergia_jar='../jAlergia/alergia.jar')

# model = run_Alergia(data=data_parsed, automaton_type='mdp', eps=0.9, print_info=True)