import json
from collections import OrderedDict
import numpy
import os

memory_action = [2, 2, 2, 2, 1]
memory_value = [([1,2]), ([-1,3]), ([0.01,0.03]), ([0.5,-0.96]), ([-0.0003254 , -0.00182071])]   
memory_policy = [([0.5,0.5]), ([0.4,0.6]), ([0.2,0.8]), ([0.9,0.1]), ([0.49965668, 0.5000472 ])]
trading_unit = [0, 98, 96, 94, 92]
results_list = []

# result_dic = {'action':memory_action[-1],'value':memory_value[-1],\
#         'policy':memory_policy[-1],'unit':trading_unit[-1]}
# # for i in range(len(memory_action)):
# #     result_dic = {'seq':(i+1),'action':memory_action[i],'value':memory_value[i],\
# #         'policy':memory_policy[i],'unit':trading_unit[i]}
# #     results_list.append(result_dic)
# print(result_dic)

result_summary = OrderedDict()
result_summary = {'action':memory_action[-1],'value':memory_value[-1],\
        'policy':memory_policy[-1],'unit':(trading_unit[-2]-trading_unit[-1])}
print(json.dumps(result_summary, ensure_ascii=False, indent="\t"))

epoch_str=1
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_path = '../../board/static/board/assets/json'
with open(os.path.join(output_path,'result_summary_{}.json'.format(epoch_str)), 'w', encoding="utf-8") as make_file:
    json.dump(result_summary, make_file, ensure_ascii=False, indent="\t")
