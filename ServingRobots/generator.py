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
            'l1': -1,
            'l2': -1,
            'r0': -1,
            'r1': -1,
            'r3': -1,
            'r7': -1,
            'r8': -1,
            'r9': -1,
            'r11': -1,
            'r12': -1,
            'r13': -1,
            'a1': -1,
            'a2': -1,
            'done': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['observations'])):
            if i == 0:
                item = {
                    'No': -1,
                    'l1': obj['observations'][i][0],
                    'l2': obj['observations'][i][1],
                    'r0': obj['observations'][i][2],
                    'r1': obj['observations'][i][3],
                    'r3': obj['observations'][i][4],
                    'r7': obj['observations'][i][5],
                    'r8': obj['observations'][i][6],
                    'r9': obj['observations'][i][7],
                    'r11': obj['observations'][i][8],
                    'r12': obj['observations'][i][9],
                    'r13': obj['observations'][i][10],
                    'a1': -1,
                    'a2': -1,
                    'done': obj['_done'][i],
                    'is_dummy': False,
                }
            else:
                item = {
                    'No': -1,
                    'l1': obj['observations'][i][0],
                    'l2': obj['observations'][i][1],
                    'r0': obj['observations'][i][2],
                    'r1': obj['observations'][i][3],
                    'r3': obj['observations'][i][4],
                    'r7': obj['observations'][i][5],
                    'r8': obj['observations'][i][6],
                    'r9': obj['observations'][i][7],
                    'r11': obj['observations'][i][8],
                    'r12': obj['observations'][i][9],
                    'r13': obj['observations'][i][10],
                    'a1': obj['actions'][i - 1][0],
                    'a2': obj['actions'][i - 1][1],
                    'done': obj['_done'][i],
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
                    elif l['l1'] == j['l1'] and l['l2'] == j['l2'] and l['r0'] == j['r0'] and l['r1'] == j['r1'] and \
                            l['r3'] == j['r3'] and l['r7'] == j['r7'] and l['r8'] == j['r8'] and l['r9'] == j['r9'] and \
                            l['r11'] == j['r11'] and l['r12'] == j['r12'] and l['r13'] == j['r13'] and \
                            l['a1'] == j['a1'] and l['a2'] == j['a2']:
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
            'module ServingRobots\n' \
            '\t\n' \
            '\tl1 : [-1..15] init 5;\n' \
            '\tl2 : [-1..15] init 10;\n' \
            '\tr0 : [-1..1] init 1;\n' \
            '\tr1 : [-1..1] init 1;\n' \
           '\tr3 : [-1..1] init 1;\n' \
           '\tr7 : [-1..1] init 1;\n' \
           '\tr8 : [-1..1] init 1;\n' \
           '\tr9 : [-1..1] init 1;\n' \
           '\tr11 : [-1..1] init 1;\n' \
           '\tr12 : [-1..1] init 1;\n' \
           '\tr13 : [-1..1] init 1;\n' \
           '\ta1 : [-1..4] init -1;\n' \
           '\ta2 : [-1..4] init -1;\n' \
           '\n'
    for state in state_list:
        data += '\t[] '
        data += '(l1 = ' + str(state['l1']) + ') & (l2 = ' + str(state['l2']) + \
                ') & (r0 = ' + str(state['r0']) + ') & (r1 = ' + str(state['r1']) + \
                ') & (r3 = ' + str(state['r3']) + ') & (r7 = ' + str(state['r7']) + \
                ') & (r8 = ' + str(state['r8']) + ') & (r9 = ' + str(state['r9']) + \
                ') & (r11 = ' + str(state['r11']) + ') & (r12 = ' + str(state['r12']) + \
                ') & (r13 = ' + str(state['r13']) + \
                ') & (a1 = ' + str(state['a1']) + ') & (a2 = ' + str(state['a2']) + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(l1\' = ' + str(next['l1']) + ') & (l2\' = ' + str(next['l2']) + \
                    ') & (r0\' = ' + str(next['r0']) + ') & (r1\' = ' + str(next['r1']) + \
                    ') & (r3\' = ' + str(next['r3']) + ') & (r7\' = ' + str(next['r7']) + \
                    ') & (r8\' = ' + str(next['r8']) + ') & (r9\' = ' + str(next['r9']) + \
                    ') & (r11\' = ' + str(next['r11']) + ') & (r12\' = ' + str(next['r12']) + \
                    ') & (r13\' = ' + str(next['r13']) + \
                    ') & (a1\' = ' + str(next['a1']) + ') & (a2\' = ' + str(next['a2']) + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(l1\' = ' + str(state['l1']) + ') & (l2\' = ' + str(state['l2']) + \
                ') & (r0\' = ' + str(state['r0']) + ') & (r1\' = ' + str(state['r1']) + \
                ') & (r3\' = ' + str(state['r3']) + ') & (r7\' = ' + str(state['r7']) + \
                ') & (r8\' = ' + str(state['r8']) + ') & (r9\' = ' + str(state['r9']) + \
                ') & (r11\' = ' + str(state['r11']) + ') & (r12\' = ' + str(state['r12']) + \
                ') & (r13\' = ' + str(state['r13']) + \
                ') & (a1\' = ' + str(state['a1']) + ') & (a2\' = ' + str(state['a2']) + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    # data += 'label \"safe\" = '
    # for state in state_list:
    #     if state['_done'] and not state['is_dummy']:
    #         data += '((r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
    #                 ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
    #                 ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
    #                 ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
    #                 ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ')) | '
    # data = data[:-3]
    # data += ';\n'
    # data += 'label \"fail\" = '
    # for state in state_list:
    #     if not state['_done'] and not state['is_dummy']:
    #         data += '((r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
    #                 ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
    #                 ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
    #                 ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
    #                 ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ')) | '
    # data = data[:-3]
    # data += ';\n'
    # for state in state_list:
    #     if not state['is_dummy']:
    #         data += 'label \"s' + str(state['No']) + '\" = ' + '(r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
    #                 ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
    #                 ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
    #                 ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
    #                 ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ');\n'
    # for state in state_list:
    #     if state['is_dummy']:
    #         data += 'label \"dummy' + str(state['No']) + '\" = ' + '(r1 = ' + str(state['r1']) + ') & (r2 = ' + str(state['r2']) + \
    #                 ') & (n0 = ' + str(state['n0']) + ') & (n1 = ' + str(state['n1']) + \
    #                 ') & (n2 = ' + str(state['n2']) + ') & (n3 = ' + str(state['n3']) + \
    #                 ') & (n4 = ' + str(state['n4']) + ') & (n5 = ' + str(state['n5']) + \
    #                 ') & (action1 = ' + str(state['action1']) + ') & (action2 = ' + str(state['action2']) + ');\n'
    # data += '\n'
    data += 'rewards "step"\n'
    data += '\t[] true : 1;\n'
    data += 'endrewards\n'
    data += 'rewards "battery"\n'
    data += '\t[] (l1 = 2) | (l1 = 4) | (l1 = 5) | (l1 = 6) | (l1 = 10) | (l1 = 14) | (l1 = 15) : 0.05;\n'
    data += '\t[] (l1 = 0) | (l1 = 1) | (l1 = 3) | (l1 = 7) | (l1 = 8) | (l1 = 9) | (l1 = 11) | (l1 = 12) | (l1 = 13) : 0.3;\n'
    data += 'endrewards\n'
    data += 'rewards "crosspath"\n'
    data += '\t[] (l1 = l2) : 0.7;\n'
    data += 'endrewards\n'
    print(data)
    f.write(data)
    f.close


timesteps = [2e4, 2.5e4, 3e4, 3.5e4, 4e4]
learning_rates = [2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
for timestep in timesteps:
    for lr in learning_rates:
        global_name = 'ppo_serving_robot' + '_' + str(timestep) + '_' + str(lr)
        lines = read_file(global_name)
        converted = convert_to_traces(lines)
        state_list = calculate_probabilities(converted)
        generate_prism_model(state_list, global_name)
