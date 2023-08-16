import copy
import os
import json


def read_file(global_name):
    f = open(global_name + '.json', 'r')
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
            'presence0': -1,
            'x0': -999,
            'y0': -999,
            'vx0': -999,
            'vy0': -999,
            'presence1': -1,
            'x1': -999,
            'y1': -999,
            'vx1': -999,
            'vy1': -999,
            'presence2': -1,
            'x2': -999,
            'y2': -999,
            'vx2': -999,
            'vy2': -999,
            'presence3': -1,
            'x3': -999,
            'y3': -999,
            'vx3': -999,
            'vy3': -999,
            'presence4': -1,
            'x4': -999,
            'y4': -999,
            'vx4': -999,
            'vy4': -999,
            'action': -1,
            '_safe': False,
            '_done': False,
            '_truncated': False,
            '_crashed': False,
            '_speed': -1,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['observations'])):
            if i == 0:
                item = {
                    'No': -1,
                    'presence0': int(obj['observations'][i][0][0]),
                    'x0': int(obj['observations'][i][0][1] * 100),
                    'y0': int(obj['observations'][i][0][2] * 100),
                    'vx0': int(obj['observations'][i][0][3] * 100),
                    'vy0': int(obj['observations'][i][0][4] * 100),
                    'presence1': int(obj['observations'][i][1][0]),
                    'x1': int(obj['observations'][i][1][1] * 100),
                    'y1': int(obj['observations'][i][1][2] * 100),
                    'vx1': int(obj['observations'][i][1][3] * 100),
                    'vy1': int(obj['observations'][i][1][4] * 100),
                    'presence2': int(obj['observations'][i][2][0]),
                    'x2': int(obj['observations'][i][2][1] * 100),
                    'y2': int(obj['observations'][i][2][2] * 100),
                    'vx2': int(obj['observations'][i][2][3] * 100),
                    'vy2': int(obj['observations'][i][2][4] * 100),
                    'presence3': int(obj['observations'][i][3][0]),
                    'x3': int(obj['observations'][i][3][1] * 100),
                    'y3': int(obj['observations'][i][3][2] * 100),
                    'vx3': int(obj['observations'][i][3][3] * 100),
                    'vy3': int(obj['observations'][i][3][4] * 100),
                    'presence4': int(obj['observations'][i][4][0]),
                    'x4': int(obj['observations'][i][4][1] * 100),
                    'y4': int(obj['observations'][i][4][2] * 100),
                    'vx4': int(obj['observations'][i][4][3] * 100),
                    'vy4': int(obj['observations'][i][4][4] * 100),
                    'action': -1,
                    '_safe': obj['_safe'][i],
                    '_done': obj['_done'][i],
                    '_truncated': obj['_truncated'][i],
                    '_crashed': obj['_crashed'][i],
                    '_speed': obj['_speed'][i],
                    'is_dummy': False,
                }
            else:
                item = {
                    'No': -1,
                    'presence0': int(obj['observations'][i][0][0]),
                    'x0': int(obj['observations'][i][0][1] * 100),
                    'y0': int(obj['observations'][i][0][2] * 100),
                    'vx0': int(obj['observations'][i][0][3] * 100),
                    'vy0': int(obj['observations'][i][0][4] * 100),
                    'presence1': int(obj['observations'][i][1][0]),
                    'x1': int(obj['observations'][i][1][1] * 100),
                    'y1': int(obj['observations'][i][1][2] * 100),
                    'vx1': int(obj['observations'][i][1][3] * 100),
                    'vy1': int(obj['observations'][i][1][4] * 100),
                    'presence2': int(obj['observations'][i][2][0]),
                    'x2': int(obj['observations'][i][2][1] * 100),
                    'y2': int(obj['observations'][i][2][2] * 100),
                    'vx2': int(obj['observations'][i][2][3] * 100),
                    'vy2': int(obj['observations'][i][2][4] * 100),
                    'presence3': int(obj['observations'][i][3][0]),
                    'x3': int(obj['observations'][i][3][1] * 100),
                    'y3': int(obj['observations'][i][3][2] * 100),
                    'vx3': int(obj['observations'][i][3][3] * 100),
                    'vy3': int(obj['observations'][i][3][4] * 100),
                    'presence4': int(obj['observations'][i][4][0]),
                    'x4': int(obj['observations'][i][4][1] * 100),
                    'y4': int(obj['observations'][i][4][2] * 100),
                    'vx4': int(obj['observations'][i][4][3] * 100),
                    'vy4': int(obj['observations'][i][4][4] * 100),
                    'action': obj['actions'][i - 1],
                    '_safe': obj['_safe'][i],
                    '_done': obj['_done'][i],
                    '_truncated': obj['_truncated'][i],
                    '_crashed': obj['_crashed'][i],
                    '_speed': obj['_speed'][i],
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
                    elif l['presence0'] == j['presence0'] and l['x0'] == j['x0'] and l['y0'] == j['y0'] and \
                            l['vx0'] == j['vx0'] and l['vy0'] == j['vy0'] and \
                            l['presence1'] == j['presence1'] and l['x1'] == j['x1'] and l['y1'] == j['y1'] and \
                            l['vx1'] == j['vx1'] and l['vy1'] == j['vy1'] and \
                            l['presence2'] == j['presence2'] and l['x2'] == j['x2'] and l['y2'] == j['y2'] and \
                            l['vx2'] == j['vx2'] and l['vy2'] == j['vy2'] and \
                            l['presence3'] == j['presence3'] and l['x3'] == j['x3'] and l['y3'] == j['y3'] and \
                            l['vx3'] == j['vx3'] and l['vy3'] == j['vy3'] and \
                            l['presence4'] == j['presence4'] and l['x4'] == j['x4'] and l['y4'] == j['y4'] and \
                            l['vx4'] == j['vx4'] and l['vy4'] == j['vy4'] and l['action'] == j['action']:
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
            'module Highway\n' \
            '\t\n' \
            '\tpresence0 : [-1..1];\n' \
            '\tx0 : [-999..100];\n' \
            '\ty0 : [-999..100];\n' \
            '\tvx0 : [-999..100];\n' \
            '\tvy0 : [-999..100];\n' \
           '\tpresence1 : [-1..1];\n' \
           '\tx1 : [-999..100];\n' \
           '\ty1 : [-999..100];\n' \
           '\tvx1 : [-999..100];\n' \
           '\tvy1 : [-999..100];\n' \
           '\tpresence2 : [-1..1];\n' \
           '\tx2 : [-999..100];\n' \
           '\ty2 : [-999..100];\n' \
           '\tvx2 : [-999..100];\n' \
           '\tvy2 : [-999..100];\n' \
           '\tpresence3 : [-1..1];\n' \
           '\tx3 : [-999..100];\n' \
           '\ty3 : [-999..100];\n' \
           '\tvx3 : [-999..100];\n' \
           '\tvy3 : [-999..100];\n' \
           '\tpresence4 : [-1..1];\n' \
           '\tx4 : [-999..100];\n' \
           '\ty4 : [-999..100];\n' \
           '\tvx4 : [-999..100];\n' \
           '\tvy4 : [-999..100];\n' \
           '\taction : [-1..10];\n' \
           '\n'
    for state in state_list:
        data += '\t[] '
        data += '(presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                ') & (vy0 = ' + str(state['vy0']) + \
                ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                ') & (vy1 = ' + str(state['vy1']) + \
                ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                ') & (vy2 = ' + str(state['vy2']) + \
                ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                ') & (vy3 = ' + str(state['vy3']) + \
                ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(presence0\' = ' + str(next['presence0']) + ') & (x0\' = ' + str(next['x0']) + \
                    ') & (y0\' = ' + str(next['y0']) + ') & (vx0\' = ' + str(next['vx0']) + \
                    ') & (vy0\' = ' + str(next['vy0']) + \
                    ') & (presence1\' = ' + str(next['presence1']) + ') & (x1\' = ' + str(next['x1']) + \
                    ') & (y1\' = ' + str(next['y1']) + ') & (vx1\' = ' + str(next['vx1']) + \
                    ') & (vy1\' = ' + str(next['vy1']) + \
                    ') & (presence2\' = ' + str(next['presence2']) + ') & (x2\' = ' + str(next['x2']) + \
                    ') & (y2\' = ' + str(next['y2']) + ') & (vx2\' = ' + str(next['vx2']) + \
                    ') & (vy2\' = ' + str(next['vy2']) + \
                    ') & (presence3\' = ' + str(next['presence3']) + ') & (x3\' = ' + str(next['x3']) + \
                    ') & (y3\' = ' + str(next['y3']) + ') & (vx3\' = ' + str(next['vx3']) + \
                    ') & (vy3\' = ' + str(next['vy3']) + \
                    ') & (presence4\' = ' + str(next['presence4']) + ') & (x4\' = ' + str(next['x4']) + \
                    ') & (y4\' = ' + str(next['y4']) + ') & (vx4\' = ' + str(next['vx4']) + \
                    ') & (vy4\' = ' + str(next['vy4']) + ') & (action\' = ' + str(next['action']) + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(presence0\' = ' + str(state['presence0']) + ') & (x0\' = ' + str(state['x0']) + \
                    ') & (y0\' = ' + str(state['y0']) + ') & (vx0\' = ' + str(state['vx0']) + \
                    ') & (vy0\' = ' + str(state['vy0']) + \
                    ') & (presence1\' = ' + str(state['presence1']) + ') & (x1\' = ' + str(state['x1']) + \
                    ') & (y1\' = ' + str(state['y1']) + ') & (vx1\' = ' + str(state['vx1']) + \
                    ') & (vy1\' = ' + str(state['vy1']) + \
                    ') & (presence2\' = ' + str(state['presence2']) + ') & (x2\' = ' + str(state['x2']) + \
                    ') & (y2\' = ' + str(state['y2']) + ') & (vx2\' = ' + str(state['vx2']) + \
                    ') & (vy2\' = ' + str(state['vy2']) + \
                    ') & (presence3\' = ' + str(state['presence3']) + ') & (x3\' = ' + str(state['x3']) + \
                    ') & (y3\' = ' + str(state['y3']) + ') & (vx3\' = ' + str(state['vx3']) + \
                    ') & (vy3\' = ' + str(state['vy3']) + \
                    ') & (presence4\' = ' + str(state['presence4']) + ') & (x4\' = ' + str(state['x4']) + \
                    ') & (y4\' = ' + str(state['y4']) + ') & (vx4\' = ' + str(state['vx4']) + \
                    ') & (vy4\' = ' + str(state['vy4']) + ') & (action\' = ' + str(state['action']) + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if state['_safe'] and not state['is_dummy']:
            data += '((presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                ') & (vy0 = ' + str(state['vy0']) + \
                ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                ') & (vy1 = ' + str(state['vy1']) + \
                ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                ') & (vy2 = ' + str(state['vy2']) + \
                ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                ') & (vy3 = ' + str(state['vy3']) + \
                ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if not state['_safe'] and not state['is_dummy']:
            data += '((presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                ') & (vy0 = ' + str(state['vy0']) + \
                ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                ') & (vy1 = ' + str(state['vy1']) + \
                ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                ') & (vy2 = ' + str(state['vy2']) + \
                ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                ') & (vy3 = ' + str(state['vy3']) + \
                ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"done\" = false'
    find = False
    for state in state_list:
        if state['_done'] and not state['is_dummy']:
            if not find:
                find = True
                data = data[:-5]
            data += '((presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                    ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                    ') & (vy0 = ' + str(state['vy0']) + \
                    ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                    ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                    ') & (vy1 = ' + str(state['vy1']) + \
                    ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                    ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                    ') & (vy2 = ' + str(state['vy2']) + \
                    ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                    ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                    ') & (vy3 = ' + str(state['vy3']) + \
                    ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                    ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                    ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ')) | '
    if find:
        data = data[:-3]
    data += ';\n'
    data += 'label \"truncated\" = false'
    find = False
    for state in state_list:
        if state['_truncated'] and not state['is_dummy']:
            if not find:
                find = True
                data = data[:-5]
            data += '((presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                    ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                    ') & (vy0 = ' + str(state['vy0']) + \
                    ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                    ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                    ') & (vy1 = ' + str(state['vy1']) + \
                    ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                    ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                    ') & (vy2 = ' + str(state['vy2']) + \
                    ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                    ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                    ') & (vy3 = ' + str(state['vy3']) + \
                    ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                    ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                    ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ')) | '
    if find:
        data = data[:-3]
    data += ';\n'
    data += 'label \"crashed\" = false'
    find = False
    for state in state_list:
        if state['_crashed'] and not state['is_dummy']:
            if not find:
                find = True
                data = data[:-5]
            data += '((presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                    ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                    ') & (vy0 = ' + str(state['vy0']) + \
                    ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                    ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                    ') & (vy1 = ' + str(state['vy1']) + \
                    ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                    ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                    ') & (vy2 = ' + str(state['vy2']) + \
                    ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                    ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                    ') & (vy3 = ' + str(state['vy3']) + \
                    ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                    ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                    ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ')) | '
    if find:
        data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                ') & (vy0 = ' + str(state['vy0']) + \
                ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                ') & (vy1 = ' + str(state['vy1']) + \
                ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                ') & (vy2 = ' + str(state['vy2']) + \
                ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                ') & (vy3 = ' + str(state['vy3']) + \
                ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                ') & (vy0 = ' + str(state['vy0']) + \
                ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                ') & (vy1 = ' + str(state['vy1']) + \
                ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                ') & (vy2 = ' + str(state['vy2']) + \
                ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                ') & (vy3 = ' + str(state['vy3']) + \
                ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ');\n'
    data += '\n'
    data += 'rewards "step"\n'
    data += '\t[] true : 1;\n'
    data += 'endrewards\n'
    data += '''rewards "lanechange"
        [] action = 0 : 1;
        [] action = 2 : 1;
    endrewards
    '''
    data += 'rewards "speed"\n'
    for state in state_list:
        data += '(presence0 = ' + str(state['presence0']) + ') & (x0 = ' + str(state['x0']) + \
                ') & (y0 = ' + str(state['y0']) + ') & (vx0 = ' + str(state['vx0']) + \
                ') & (vy0 = ' + str(state['vy0']) + \
                ') & (presence1 = ' + str(state['presence1']) + ') & (x1 = ' + str(state['x1']) + \
                ') & (y1 = ' + str(state['y1']) + ') & (vx1 = ' + str(state['vx1']) + \
                ') & (vy1 = ' + str(state['vy1']) + \
                ') & (presence2 = ' + str(state['presence2']) + ') & (x2 = ' + str(state['x2']) + \
                ') & (y2 = ' + str(state['y2']) + ') & (vx2 = ' + str(state['vx2']) + \
                ') & (vy2 = ' + str(state['vy2']) + \
                ') & (presence3 = ' + str(state['presence3']) + ') & (x3 = ' + str(state['x3']) + \
                ') & (y3 = ' + str(state['y3']) + ') & (vx3 = ' + str(state['vx3']) + \
                ') & (vy3 = ' + str(state['vy3']) + \
                ') & (presence4 = ' + str(state['presence4']) + ') & (x4 = ' + str(state['x4']) + \
                ') & (y4 = ' + str(state['y4']) + ') & (vx4 = ' + str(state['vx4']) + \
                ') & (vy4 = ' + str(state['vy4']) + ') & (action = ' + str(state['action']) + ') : ' + \
                str(state['_speed']) + ';\n'
    data += 'endrewards\n'

    print(data)
    f.write(data)
    f.close


timesteps = [1e4, 1.5e4, 2e4, 2.5e4, 3e4]
learning_rates = [2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
for timestep in timesteps:
    for lr in learning_rates:
        global_name = 'ppo_merge_v0' + '_' + str(timestep) + '_' + str(lr)
        lines = read_file(global_name)
        converted = convert_to_traces(lines)
        state_list = calculate_probabilities(converted)
        generate_prism_model(state_list, global_name)
