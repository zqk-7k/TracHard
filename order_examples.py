from acc_list40000_rand import *


acc_list = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(epoch0_acc)):
    acc_list[0].append(epoch0_acc[i])
    acc_list[1].append(epoch1_acc[i])
    acc_list[2].append(epoch2_acc[i])
    acc_list[3].append(epoch3_acc[i])
    acc_list[4].append(epoch4_acc[i])
    acc_list[5].append(epoch5_acc[i])
    acc_list[6].append(epoch6_acc[i])
    acc_list[7].append(epoch7_acc[i])
    acc_list[8].append(epoch8_acc[i])
    acc_list[9].append(epoch9_acc[i])


'''
把不容易忘记的那些数据可以删去，跟上一个epoch比，
若上个为1，这次为1，则score加2，若上次为1，这次为0，则score加1
若上个为0，这次为0，则score加0，若上次为0，这次为1，则score加1
'''
'''
这里虽然计分方式不同，但实际hard还是采取的所有都对的情况，easy采取的所有都错的情况，所以就没改动，可以忽略这种方式，直接把错误为0，正确为1
'''


#计算得分值，返回一个list
def get_score(acc_list,forget_epochnum):
    score_list = []
    for i in range(1, forget_epochnum):
        for j in range(len(acc_list[i])):
            if (i == 1):
                if (acc_list[i][j] + acc_list[i - 1][j] == 1):
                    score_list.append(1)
                elif (acc_list[i][j] + acc_list[i - 1][j] == 2):
                    score_list.append(2)
                else:
                    score_list.append(0)
            else:
                if (acc_list[i][j] + acc_list[i - 1][j] == 1):
                    score_list[j] += 1
                elif (acc_list[i][j] + acc_list[i - 1][j] == 2):
                    score_list[j] += 2
    return score_list

#获取难以忘记数据的对应编号
#score_list为得分list，get_percent取出难以忘记数据占总数量的百分比
def get_unforget_num(score_list,get_percent):
    get_num = get_percent * len(score_list)  #取出的数量
    index = int(len(score_list) - get_num)   #排序从小到大，对应的索引
    unforget_num = []
    score_copy = []
    for i in range(len(score_list)):
        score_copy.append(score_list[i])
    score_copy.sort()
    for i in range(len(score_list)):
        if(score_list[i] >= score_copy[index]):
            unforget_num.append(i)
            if(get_num == len(unforget_num)):
                return unforget_num

#forget的读取的epoch总数量
forget_epochnum = 10
score_list = get_score(acc_list,forget_epochnum)

num = 0
difficult_num = []
for i in range(len(score_list)):
    if(score_list[i] == 18):
        difficult_num.append(i)
        num += 1
# print(len(difficult_num))

#
# num_account = [0,0,0,0,0,0,0,0,0,0]
# difficult_data = [[],[],[],[],[],[],[],[],[],[]]

# unforget_num = get_unforget_num(score_list,0.5)

