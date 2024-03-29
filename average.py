import json
import numpy as np
prefix = '/home/zxp/code/PL-Marker-master/result/re/'
resultfilename = '/results.json'
f1s = []
dataset = '14res_sim'

f1s = []
ps = []
rs = []
singel_f1s = []
singel_ps = []
singel_rs = []
multi_f1s = []
multi_ps = []
multi_rs = []
for i in range(41, 46):
    best_f1 = 0
    best_p = 0
    best_r = 0

    fileename = prefix + dataset + f'/final_best_result4deepke_{i}'+resultfilename

    f1 = json.load(open(fileename))
    for key,item in f1.items():
        if key.startswith('f1_'):
            if item>best_f1:
                best_f1 = item
                end = key[3:]
                best_p = f1['p_'+end]
                best_r = f1['r_'+end]
                # best_singel_p = f1['singel_p_'+end]
                # best_singel_r = f1['singel_r_'+end]
                # best_singel_f1 = f1['singel_f1_'+end]
                # best_multi_p = f1['multi_p_'+end]
                # best_multi_r = f1['multi_r_'+end]
                # best_multi_f1 = f1['multi_f1_'+end]
    print (fileename)
    f1s.append(best_f1)
    ps.append(best_p)
    rs.append(best_r)

    # singel_f1s.append(best_singel_f1)
    # singel_ps.append(best_singel_p)
    # singel_rs.append(best_singel_r)
    #
    # multi_f1s.append(best_multi_f1)
    # multi_ps.append(best_multi_p)
    # multi_rs.append(best_multi_r)


print ('F1:')
print (f1s)
f1s = np.array(f1s)
print (np.mean(f1s)*100)
print (np.std(f1s)*100)
print ('p:')
print (ps)
ps = np.array(ps)
print (np.mean(ps)*100)
print (np.std(ps)*100)
print ('r:')
print (rs)
rs = np.array(rs)
print (np.mean(rs)*100)
print (np.std(rs)*100)

# print ('=======================================================')
# print ('singel_F1:')
# print (singel_f1s)
# f1s = np.array(singel_f1s)
# print (np.mean(f1s)*100)
# print (np.std(f1s)*100)
# print ('singel_p:')
# print (singel_ps)
# ps = np.array(singel_ps)
# print (np.mean(ps)*100)
# print (np.std(ps)*100)
# print ('singel_r:')
# print (singel_rs)
# rs = np.array(singel_rs)
# print (np.mean(rs)*100)
# print (np.std(rs)*100)
#
# print ('=======================================================')
# print ('multi_F1:')
# print (multi_f1s)
# f1s = np.array(multi_f1s)
# print (np.mean(f1s)*100)
# print (np.std(f1s)*100)
# print ('multi_p:')
# print (multi_ps)
# ps = np.array(multi_ps)
# print (np.mean(ps)*100)
# print (np.std(ps)*100)
# print ('multi_r:')
# print (multi_rs)
# rs = np.array(multi_rs)
# print (np.mean(rs)*100)
# print (np.std(rs)*100)