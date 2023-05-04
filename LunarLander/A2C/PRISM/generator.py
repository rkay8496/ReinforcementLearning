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
            'x': 0,
            'y': 0,
            'xv': 0,
            'yv': 0,
            'a': 0,
            'av': 0,
            'l': 0,
            'r': 0,
            'not': False,
            'le': False,
            'me': False,
            're': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['x'])):
            item = {
                'No': -1,
                'x': int(obj['x'][i][1] * 100),
                'y': int(obj['y'][i][1] * 100),
                'xv': int(obj['xv'][i][1] * 100),
                'yv': int(obj['yv'][i][1] * 100),
                'a': int(obj['a'][i][1] * 100),
                'av': int(obj['av'][i][1] * 100),
                'l': int(obj['l'][i][1] * 100),
                'r': int(obj['r'][i][1] * 100),
                'not': obj['not'][i][1],
                'le': obj['le'][i][1],
                'me': obj['me'][i][1],
                're': obj['re'][i][1],
                'is_dummy': False
            }
            path.append(item)
        if obj['aux0']:
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
                    elif l['x'] == j['x'] and l['y'] == j['y'] and \
                            l['xv'] == j['xv'] and \
                            l['yv'] == j['yv'] and l['a'] == j['a'] and \
                            l['av'] == j['av'] and \
                            l['l'] == j['l'] and l['r'] == j['r'] and \
                            l['not'] == j['not'] and l['le'] == j['le'] and l['me'] == j['me'] and \
                            l['re'] == j['re']:
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
    f = open('./dtmc.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module LunarLander\n' \
            '\t\n' \
            '\tx : [-150..150];\n' \
            '\ty : [-150..150];\n' \
            '\txv : [-500..500];\n' \
            '\tyv : [-500..500];\n' \
            '\ta : [-314..314];\n' \
            '\tav : [-500..500];\n' \
            '\tl : [0..100];\n' \
            '\tr : [0..100];\n' \
            '\tnot : bool init true;\n' \
            '\tle : bool init false;\n' \
            '\tme : bool init false;\n' \
            '\tre : bool init false;\n' \
           '\n'
    for state in state_list:
        data += '\t[] '
        data += '(x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (xv = ' + str(state['xv']) + \
                ') & (yv = ' + str(state['yv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (not = ' + str(state['not']).lower() + ') & (le = ' + str(state['le']).lower() + \
                ') & (me = ' + str(state['me']).lower() + ') & (re = ' + str(state['re']).lower() + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(x\' = ' + str(next['x']) + ') & (y\' = ' + str(next['y']) + \
                        ') & (xv\' = ' + str(next['xv']) + \
                        ') & (yv\' = ' + str(next['yv']) + ') & (a\' = ' + str(next['a']) + \
                        ') & (av\' = ' + str(next['av']) + \
                        ') & (l\' = ' + str(next['l']) + ') & (r\' = ' + str(next['r']) + \
                        ') & (not\' = ' + str(next['not']).lower() + ') & (le\' = ' + str(next['le']).lower() + \
                        ') & (me\' = ' + str(next['me']).lower() + ') & (re\' = ' + str(next['re']).lower() + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(x\' = ' + str(state['x']) + ') & (y\' = ' + str(state['y']) +  \
                    ') & (xv\' = ' + str(state['xv']) + \
                    ') & (yv\' = ' + str(state['yv']) + ') & (a\' = ' + str(state['a']) + \
                    ') & (av\' = ' + str(state['av']) + \
                    ') & (l\' = ' + str(state['l']) + ') & (r\' = ' + str(state['r']) + \
                    ') & (not\' = ' + str(state['not']).lower() + ') & (le\' = ' + str(state['le']).lower() + \
                    ') & (me\' = ' + str(state['me']).lower() + ') & (re\' = ' + str(state['re']).lower() + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if len(state['next']) > 0:
            data += '((x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (xv = ' + str(state['xv']) + \
                ') & (yv = ' + str(state['yv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (not = ' + str(state['not']).lower() + ') & (le = ' + str(state['le']).lower() + \
                ') & (me = ' + str(state['me']).lower() + ') & (re = ' + str(state['re']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if len(state['next']) == 1 and state['next'][0]['is_dummy']:
            data += '((x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (xv = ' + str(state['xv']) + \
                ') & (yv = ' + str(state['yv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (not = ' + str(state['not']).lower() + ') & (le = ' + str(state['le']).lower() + \
                ') & (me = ' + str(state['me']).lower() + ') & (re = ' + str(state['re']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (xv = ' + str(state['xv']) + \
                ') & (yv = ' + str(state['yv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (not = ' + str(state['not']).lower() + ') & (le = ' + str(state['le']).lower() + \
                ') & (me = ' + str(state['me']).lower() + ') & (re = ' + str(state['re']).lower() + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (xv = ' + str(state['xv']) + \
                ') & (yv = ' + str(state['yv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (not = ' + str(state['not']).lower() + ') & (le = ' + str(state['le']).lower() + \
                ') & (me = ' + str(state['me']).lower() + ') & (re = ' + str(state['re']).lower() + ');\n'
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
