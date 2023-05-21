import copy
import os
import json


def read_file():
    f = open(os.getcwd() + '/../traces_01.json', 'r')
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
            'state': -1,
            'action': -1,
            '_safe': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['state'])):
            if i == 0:
                item = {
                    'No': -1,
                    'state': obj['state'][i][1],
                    'action': -1,
                    '_safe': obj['_safe'][i][1],
                    'is_dummy': False,
                }
            else:
                item = {
                    'No': -1,
                    'state': obj['state'][i][1],
                    'action': obj['action'][i - 1][1],
                    '_safe': obj['_safe'][i][1],
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
                    elif l['state'] == j['state'] and l['action'] == j['action']:
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


def generate_prism_model(state_list):
    f = open('dtmc.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module NuclearReactorTrip\n' \
            '\t\n' \
            '\tstate : [-1..3];\n' \
            '\taction : [-1..3];\n' \
           '\n'
    for state in state_list:
        data += '\t[] '
        data += '(state = ' + str(state['state']) + ') & (action = ' + str(state['action']) + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(state\' = ' + str(next['state']) + ') & (action\' = ' + str(next['action']) + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(state\' = ' + str(state['state']) + ') & (action\' = ' + str(state['action']) + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    # 검토해봐야 하는 코드
    data += 'label \"safe\" = '
    for state in state_list:
        if state['_safe'] and not state['is_dummy']:
            data += '((state = ' + str(state['state']) + ') & (action = ' + str(state['action']) + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if not state['_safe'] and not state['is_dummy']:
            data += '((state = ' + str(state['state']) + ') & (action = ' + str(state['action']) + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(state = ' + str(state['state']) + ') & (action = ' + str(state['action']) + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(state = ' + str(state['state']) + ') & (action = ' + str(state['action']) + ');\n'
    data += '\n'
    data += 'rewards "step"\n'
    data += '\t[] true : 1;\n'
    data += 'endrewards'
    print(data)
    f.write(data)
    f.close


lines = read_file()
converted = convert_to_traces(lines)
state_list = calculate_probabilities(converted)
generate_prism_model(state_list)
