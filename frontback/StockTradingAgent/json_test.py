import json
from collections import OrderedDict
import numpy
import pandas
memory_action = [2, 2, 2, 2, 1]
memory_value = [([1,2]), ([-1,3]), ([0.01,0.03]), ([0.5,-0.96]), ([-0.0003254 , -0.00182071])]   
memory_policy = [([0.5,0.5]), ([0.4,0.6]), ([0.2,0.8]), ([0.9,0.1]), ([0.49965668, 0.5000472 ])]
trading_unit = [0, 98, 96, 94, 92]
results_list = []

for i in range(len(memory_action)):
    result_dic = {'seq':(i+1),'action':memory_action[i],'value':memory_value[i],\
        'policy':memory_policy[i],'unit':trading_unit[i]}
    results_list.append(result_dic)
# print(results_list)

result_summary = OrderedDict()
result_summary = results_list
# print(json.dumps(result_summary, ensure_ascii=False, indent="\t"))
with open(os.path.join(test_dir,'result_summary_{}.json'.format(epoch_str), 'w', encoding="utf-8") as make_file:
    json.dump(result_summary, make_file, ensure_ascii=False, indent="\t")
