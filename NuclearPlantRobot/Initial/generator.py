import copy
import os
import json


def read_file(global_name):
    f = open('./models/' + global_name + '.json', 'r')
    lines = f.readlines()
    f.close()
    return lines


def convert_to_traces(lines):
    converted = []
    for line in lines:
        obj = json.loads(line)
        path = []
        dummy = {
            'No': -1,
            'r1': -1,
            'r2': -1,
            'n0': -1,
            'n1': -1,
            'n2': -1,
            'n3': -1,
            'n4': -1,
            'n5': -1,
            'action1': -1,
            'action2': -1,
            '_safe': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['observations'])):
            if i == 0:
                item = {
                    'No': -1,
                    'r1': obj['observations'][i][0],
                    'r2': obj['observations'][i][1],
                    'n0': obj['observations'][i][2],
                    'n1': obj['observations'][i][3],
                    'n2': obj['observations'][i][4],
                    'n3': obj['observations'][i][5],
                    'n4': obj['observations'][i][6],
                    'n5': obj['observations'][i][7],
                    'action1': -1,
                    'action2': -1,
                    '_safe': obj['_safe'][i],
                    'is_dummy': False,
                }
            else:
                item = {
                    'No': -1,
                    'r1': obj['observations'][i][0],
                    'r2': obj['observations'][i][1],
                    'n0': obj['observations'][i][2],
                    'n1': obj['observations'][i][3],
                    'n2': obj['observations'][i][4],
                    'n3': obj['observations'][i][5],
                    'n4': obj['observations'][i][6],
                    'n5': obj['observations'][i][7],
                    'action1': obj['actions'][i - 1][0],
                    'action2': obj['actions'][i - 1][1],
                    '_safe': obj['_safe'][i],
                    'is_dummy': False,
                }
            path.append(item)
            path.append(dummy)
            path.append(item)
        converted.append(path)
    num = 0
    find = False
    for i in converted:
        for j in i:
            if j['No'] == -1:
                j['No'] = num
                num += 1
            for k in converted:
                for l in k:
                    if l['is_dummy'] and j['is_dummy']:
                        l['No'] = j['No']
                    elif l['r1'] == j['r1'] and l['r2'] == j['r2'] and l['n0'] == j['n0'] and l['n1'] == j['n1'] and \
                            l['n2'] == j['n2'] and l['n3'] == j['n3'] and l['n4'] == j['n4'] and l['n5'] == j['n5'] and \
                            l['action1'] == j['action1'] and l['action2'] == j['action2']:
                        l['No'] = j['No']
    return converted


def calculate_probabilities(converted):
    state_list = []
    for path in converted:
        for state in path:
            state['next'] = []
            if len([s for s in state_list if s['No'] == state['No']]) == 0:
                state_list.append(state)
    for state in state_list:
        for path in converted:
            for idx, step in enumerate(path):
                if state['No'] == step['No']:
                    if idx < len(path) - 1:
                        find = False
                        for s in state['next']:
                            if s['No'] == path[idx + 1]['No']:
                                s['num_of_occurrence'] += 1
                                s['probability'] = ''
                                find = True
                                break
                            else:
                                find = False
                        if not find:
                            obj = copy.deepcopy(path[idx + 1])
                            del obj['next']
                            obj['num_of_occurrence'] = 1
                            obj['probability'] = ''
                            state['next'].append(obj)
    for state in state_list:
        total_occurrence = 0
        for s in state['next']:
            total_occurrence += s['num_of_occurrence']
        for s in state['next']:
            s['probability'] = str(s['num_of_occurrence']) + '/' + str(total_occurrence)
    for state in state_list:
        print(state)
    return state_list


def generate_prism_model(state_list, global_name):
    f = open(global_name + '.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module MovingObstacle\n' \
            '\t\n' \
            '\tr1 : [-1..5];\n' \
            '\tr2 : [-1..5];\n' \
            '\tn0 : [-1..3];\n' \
            '\tn1 : [-1..3];\n' \
           '\tn2 : [-1..3];\n' \
           '\tn3 : [-1..3];\n' \
           '\tn4 : [-1..3];\n' \
           '\tn5 : [-1..3];\n' \
           '\taction1 : [-1..21];\n' \
           '\taction2 : [-1..21];\n' \
           '\n'
    for state in state_list:
        data += '\t[] '
        data += '(r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
                ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
                ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
                ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
                ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(r1\' = ' + str(next['r1']) + ') & (r2\' = ' + str(next['r2']) + \
                        ') & (n0\' = ' + str(next['n0']) + ') & (n1\' = ' + str(next['n1']) + \
                        ') & (n2\' = ' + str(next['n2']) + ') & (n3\' = ' + str(next['n3']) + \
                        ') & (n4\' = ' + str(next['n4']) + ') & (n5\' = ' + str(next['n5']) + \
                        ') & (action1\' = ' + str(next['action1']) + ') & (action2\' = ' + str(next['action2']) + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(r1\' = ' + str(state['r1']) + ') & (r2\' = ' + str(state['r2']) +  \
                    ') & (n0\' = ' + str(state['n0']) + ') & (n1\' = ' + str(state['n1']) + \
                    ') & (n2\' = ' + str(state['n2']) + ') & (n3\' = ' + str(state['n3']) + \
                    ') & (n4\' = ' + str(state['n4']) + ') & (n5\' = ' + str(state['n5']) + \
                    ') & (action1\' = ' + str(state['action1']) + ') & (action2\' = ' + str(state['action2']) + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if state['_safe'] and not state['is_dummy']:
            data += '((r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
                    ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
                    ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
                    ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
                    ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if not state['_safe'] and not state['is_dummy']:
            data += '((r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
                    ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
                    ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
                    ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
                    ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
                    ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
                    ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
                    ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
                    ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
                    ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
                    ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
                    ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
                    ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ');\n'
    data += '\n'
    data += 'rewards "step"\n'
    data += '\t[] true : 1;\n'
    data += 'endrewards\n'
    print(data)
    f.write(data)
    f.close

#     rewards
#     "battery"
#     [](action1=0): 3;
#     [](action1=1): 1.5;
#     [](action1=2): 2;
#     [](action1=3): 3;
#     [](action1=4): 1.5;
#     [](action1=5): 3.5;
#     [](action1=6): 7;
#     [](action1=7): 1.5;
#     [](action1=8): 5.5;
#     [](action1=9): 2;
#     [](action1=10): 1.5;
#     [](action1=11): 3.5;
#     [](action1=12): 7;
#     [](action1=13): 8;
#     [](action1=14): 8;
#     [](action1=15): 7;
#     [](action1=16): 7;
#     [](action1=17): 8;
#     [](action1=18): 6;
#     [](action1=19): 5.5;
#     [](action1=20): 8;
#     [](action1=21): 6;
#     [](action2=0): 3;
#     [](action2=1): 1.5;
#     [](action2=2): 2;
#     [](action2=3): 3;
#     [](action2=4): 1.5;
#     [](action2=5): 3.5;
#     [](action2=6): 7;
#     [](action2=7): 1.5;
#     [](action2=8): 5.5;
#     [](action2=9): 2;
#     [](action2=10): 1.5;
#     [](action2=11): 3.5;
#     [](action2=12): 7;
#     [](action2=13): 8;
#     [](action2=14): 8;
#     [](action2=15): 7;
#     [](action2=16): 7;
#     [](action2=17): 8;
#     [](action2=18): 6;
#     [](action2=19): 5.5;
#     [](action2=20): 8;
#     [](action2=21): 6;
#
#
# endrewards
#
# rewards
# "exposure"
# [](r1=4): 0.04;
# [](r2=4): 0.04
# endrewards


settings = [
    (6e4, 4e-4), (6e4, 6e-4), (6e4, 8e-4), (6e4, 1e-3),
    (8e4, 2e-4), (8e4, 4e-4), (8e4, 6e-4), (8e4, 8e-4), (8e4, 1e-3),
    (1e5, 2e-4), (1e5, 4e-4), (1e5, 6e-4), (1e5, 8e-4), (1e5, 1e-3),
]
for setting in settings:
    global_name = 'ppo_nuclear_plant_robot' + '_' + str(setting[0]) + '_' + str(setting[1])
    lines = read_file(global_name)
    converted = convert_to_traces(lines)
    state_list = calculate_probabilities(converted)
    generate_prism_model(state_list, global_name)
