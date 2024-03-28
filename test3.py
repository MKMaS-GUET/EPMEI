import argparse
import json

#变成列表形式  #只处理ner模型出来的结果（ner模型出来的结果不需要改区间）
for dataset in ['14res']:
    for t in ['test']:

        # file_path = f'/home/zxp/code/PL-Marker-master/result/nerOld/{dataset}_newdata/ent_pred_{t}.json'
        file_path = f'/home/zxp/code/PL-Marker-master/result/nerOld/ate_ope_mpl14res_newdata128/ent_pred_{t}.json'
        # file_path2 = f'/home/zxp/code/PL-Marker-master/result/nerOld/{dataset}_newdata/ent_pred_{t}_pro.json'
        file_path2 = f'/home/zxp/code/PL-Marker-master/result/nerOld/ate_ope_mpl14res_newdata128/ent_pred_{t}_pro.json'


        print(type(file_path))
        list = []
        file = open(file_path, 'r', encoding = 'utf-8')

        for id,line in enumerate(file):


            data = json.loads(line)
            list.append(data)

        with open(file_path2,'w',encoding='utf-8') as f:
            json.dump(list,f,ensure_ascii=False)



