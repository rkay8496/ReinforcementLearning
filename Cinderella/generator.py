import copy
import os
import json


def read_file():
    f = open('./ppo_cinderella_5_10_5_2.json', 'r')
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
            'b1': -1,
            'b2': -1,
            'b3': -1,
            'b4': -1,
            'b5': -1,
            'c1': -1,
            'c2': -1,
            'c3': -1,
            'c4': -1,
            'c5': -1,
            '_safe': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['observations'])):
            if i == 0:
                item = {
                    'No': -1,
                    'b1': obj['observations'][i][0],
                    'b2': obj['observations'][i][1],
                    'b3': obj['observations'][i][2],
                    'b4': obj['observations'][i][3],
                    'b5': obj['observations'][i][4],
                    'c1': -1,
                    'c2': -1,
                    'c3': -1,
                    'c4': -1,
                    'c5': -1,
                    '_safe': obj['_safe'][i],
                    'is_dummy': False,
                }
            else:
                item = {
                    'No': -1,
                    'b1': obj['observations'][i][0],
                    'b2': obj['observations'][i][1],
                    'b3': obj['observations'][i][2],
                    'b4': obj['observations'][i][3],
                    'b5': obj['observations'][i][4],
                    'c1': obj['actions'][i - 1][0],
                    'c2': obj['actions'][i - 1][1],
                    'c3': obj['actions'][i - 1][2],
                    'c4': obj['actions'][i - 1][3],
                    'c5': obj['actions'][i - 1][4],
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
                    elif l['b1'] == j['b1'] and l['b2'] == j['b2'] and l['b3'] == j['b3'] and l['b4'] == j['b4'] and \
                            l['b5'] == j['b5'] and l['c1'] == j['c1'] and l['c2'] == j['c2'] and l['c3'] == j['c3'] and \
                            l['c4'] == j['c4'] and l['c5'] == j['c5']:
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
    f = open('ppo_cinderella_5_10_5_2.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module Cinderella\n' \
            '\t\n' \
            '\tb1 : [-1..99];\n' \
            '\tb2 : [-1..99];\n' \
            '\tb3 : [-1..99];\n' \
            '\tb4 : [-1..99];\n' \
            '\tb5 : [-1..99];\n' \
            '\tc1 : [-1..99];\n' \
            '\tc2 : [-1..99];\n' \
            '\tc3 : [-1..99];\n' \
            '\tc4 : [-1..99];\n' \
            '\tc5 : [-1..99];\n' \
            '\n'
    for state in state_list:
        data += '\t[] '
        data += '(b1 = ' + str(state['b1']) + ') & (b2 = ' + str(state['b2']) + \
                ') & (b3 = ' + str(state['b3']) + ') & (b4 = ' + str(state['b4']) + \
                ') & (b5 = ' + str(state['b5']) + ') & (c1 = ' + str(state['c1']) + \
                ') & (c2 = ' + str(state['c2']) + ') & (c3 = ' + str(state['c3']) + \
                ') & (c4 = ' + str(state['c4']) + ') & (c5 = ' + str(state['c5']) + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(b1\' = ' + str(next['b1']) + ') & (b2\' = ' + str(next['b2']) + \
                        ') & (b3\' = ' + str(next['b3']) + ') & (b4\' = ' + str(next['b4']) + \
                        ') & (b5\' = ' + str(next['b5']) + ') & (c1\' = ' + str(next['c1']) + \
                        ') & (c2\' = ' + str(next['c2']) + ') & (c3\' = ' + str(next['c3']) + \
                        ') & (c4\' = ' + str(next['c4']) + ') & (c5\' = ' + str(next['c5']) + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(b1\' = ' + str(state['b1']) + ') & (b2\' = ' + str(state['b2']) +  \
                    ') & (b3\' = ' + str(state['b3']) + ') & (b4\' = ' + str(state['b4']) + \
                    ') & (b5\' = ' + str(state['b5']) + ') & (c1\' = ' + str(state['c1']) + \
                    ') & (c2\' = ' + str(state['c2']) + ') & (c3\' = ' + str(state['c3']) + \
                    ') & (c4\' = ' + str(state['c4']) + ') & (c5\' = ' + str(state['c5']) + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if state['_safe'] and not state['is_dummy']:
            data += '((b1 = ' + str(state['b1']) + ') & (b2 = ' + str(state['b2']) + \
                ') & (b3 = ' + str(state['b3']) + ') & (b4 = ' + str(state['b4']) + \
                ') & (b5 = ' + str(state['b5']) + ') & (c1 = ' + str(state['c1']) + \
                ') & (c2 = ' + str(state['c2']) + ') & (c3 = ' + str(state['c3']) + \
                ') & (c4 = ' + str(state['c4']) + ') & (c5 = ' + str(state['c5']) + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if not state['_safe'] and not state['is_dummy']:
            data += '((b1 = ' + str(state['b1']) + ') & (b2 = ' + str(state['b2']) + \
                ') & (b3 = ' + str(state['b3']) + ') & (b4 = ' + str(state['b4']) + \
                ') & (b5 = ' + str(state['b5']) + ') & (c1 = ' + str(state['c1']) + \
                ') & (c2 = ' + str(state['c2']) + ') & (c3 = ' + str(state['c3']) + \
                ') & (c4 = ' + str(state['c4']) + ') & (c5 = ' + str(state['c5']) + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(b1 = ' + str(state['b1']) + ') & (b2 = ' + str(state['b2']) + \
                ') & (b3 = ' + str(state['b3']) + ') & (b4 = ' + str(state['b4']) + \
                ') & (b5 = ' + str(state['b5']) + ') & (c1 = ' + str(state['c1']) + \
                ') & (c2 = ' + str(state['c2']) + ') & (c3 = ' + str(state['c3']) + \
                ') & (c4 = ' + str(state['c4']) + ') & (c5 = ' + str(state['c5']) + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(b1 = ' + str(state['b1']) + ') & (b2 = ' + str(state['b2']) + \
                ') & (b3 = ' + str(state['b3']) + ') & (b4 = ' + str(state['b4']) + \
                ') & (b5 = ' + str(state['b5']) + ') & (c1 = ' + str(state['c1']) + \
                ') & (c2 = ' + str(state['c2']) + ') & (c3 = ' + str(state['c3']) + \
                ') & (c4 = ' + str(state['c4']) + ') & (c5 = ' + str(state['c5']) + ');\n'
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
