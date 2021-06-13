import numpy as np
from cec2013.cec2013 import *
from modules2 import *
import argparse
import time

def run(func_id):
    f = CEC2013(func_id)
    D = f.get_dimension()
    pop_size = 16 * D
    archive = np.empty((0, D))
    evaluation_remaind_cnt = f.get_maxfes()

    taboo_points = np.empty((0, D))
    radius_of_taboo_points = np.array([])
    while evaluation_remaind_cnt > 0:
        population = initialization(f, 4 * pop_size, taboo_points, radius_of_taboo_points)
        evaluation_remaind_cnt -= population.shape[0]
        score = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            score[i] = f.evaluate(population[i])
        id_sorted_pop = np.argsort(score)[::-1]
        population = population[id_sorted_pop[:1 * pop_size]]# decrese 
        tmp_taboo_point = np.empty((0, D)) # use to store the best individuals
        tmp_radius_of_taboo_points = np.array([])

        start_time = time.time()*1000

        for individual in population:
            new_population, evaluation_remaind_cnt = cmsa(np.reshape(individual, (-1, D)), evaluation_remaind_cnt, f, np.ones(1) * f.evaluate(individual),
                                       taboo_points, radius_of_taboo_points)
            radius_of_this_taboo = np.linalg.norm(new_population[0] - individual, 2)
            tmp_radius_of_taboo_points = np.hstack((tmp_radius_of_taboo_points, radius_of_this_taboo))
            tmp_taboo_point = np.vstack((tmp_taboo_point, new_population[0]))

            archive, evaluation_remaind_cnt = union(archive, new_population, f, evaluation_remaind_cnt, start_time)
        taboo_points = tmp_taboo_point
        radius_of_taboo_points = tmp_radius_of_taboo_points
    return archive

def experiment():
    runcnt = 50
    problemcnt = 20+1
    recall1 = np.zeros((problemcnt-1,runcnt))
    recall2 = np.zeros((problemcnt-1,runcnt))
    recall3 = np.zeros((problemcnt-1,runcnt))
    recall4 = np.zeros((problemcnt-1,runcnt))
    recall5 = np.zeros((problemcnt-1,runcnt))
    precison1 = np.zeros((problemcnt-1,runcnt))
    precison2 = np.zeros((problemcnt-1,runcnt))
    precison3 = np.zeros((problemcnt-1,runcnt))
    precison4 = np.zeros((problemcnt-1,runcnt))
    precison5 = np.zeros((problemcnt-1,runcnt))
    f1_stasic1 = np.zeros((problemcnt-1,runcnt))
    f1_stasic2 = np.zeros((problemcnt-1,runcnt))
    f1_stasic3 = np.zeros((problemcnt-1,runcnt))
    f1_stasic4 = np.zeros((problemcnt-1,runcnt))
    f1_stasic5 = np.zeros((problemcnt-1,runcnt))

    for exp in range(runcnt):
        for func_id in range(1, problemcnt):
            print('Prblem', func_id, 'RUN', exp)
            # ans = main4(func_id)
            ans = run(func_id)
            recall,precison,f1_stasic = print_result(ans[:, :-2], CEC2013(func_id),func_id,exp)
            recall1[func_id-1][exp] = recall[0]
            recall2[func_id-1][exp] = recall[1]
            recall3[func_id-1][exp] = recall[2]
            recall4[func_id-1][exp] = recall[3]
            recall5[func_id-1][exp] = recall[4]
            precison1[func_id-1][exp] = precison[0]
            precison2[func_id-1][exp] = precison[1]
            precison3[func_id-1][exp] = precison[2]
            precison4[func_id-1][exp] = precison[3]
            precison5[func_id-1][exp] = precison[4]
            f1_stasic1[func_id-1][exp] = precison[0]
            f1_stasic2[func_id-1][exp] = precison[1]
            f1_stasic3[func_id-1][exp] = precison[2]
            f1_stasic4[func_id-1][exp] = precison[3]
            f1_stasic5[func_id-1][exp] = precison[4]


            score = np.zeros(ans.shape[0])
            f = CEC2013(func_id)
            characters1 = np.empty([ans.shape[0],1], dtype = str)
            characters2 = np.empty([ans.shape[0],1], dtype = str)

            for i in range(ans.shape[0]):
                score[i] = f.evaluate(ans[i, :-2])
                characters1[i,0] = '='
                characters2[i,0] = '@'
            adds = np.ones(ans.shape[0],dtype= 'int8')
             
            anss = np.hstack((ans[:,:-2], characters1.reshape(ans.shape[0], 1), score.reshape(ans.shape[0], 1),\
             characters2.reshape(ans.shape[0], 1), ans[:,-2].reshape(ans.shape[0], 1),  ans[:,-1].reshape(ans.shape[0], 1), adds.reshape(ans.shape[0],1)))
            np.savetxt('./results/' + 'problem%03d' % (func_id) + 'run%03d' % (exp+1) + '.dat', anss,fmt = '%s')
   
 
    # save data to csv
    import pandas as pd 
    for func_id in range(1, problemcnt):
        recordeddata = np.zeros((runcnt,3*5))
        recordeddata[:,0] = recall1[func_id-1][:]
        recordeddata[:,1] = recall2[func_id-1][:]
        recordeddata[:,2] = recall3[func_id-1][:]  
        recordeddata[:,3] = recall4[func_id-1][:]
        recordeddata[:,4] = recall5[func_id-1][:]

        recordeddata[:,5] = precison1[func_id-1][:]
        recordeddata[:,6] = precison2[func_id-1][:]
        recordeddata[:,7] = precison3[func_id-1][:]  
        recordeddata[:,8] = precison4[func_id-1][:]
        recordeddata[:,9] = precison5[func_id-1][:]

        recordeddata[:,10] = f1_stasic1[func_id-1][:]
        recordeddata[:,11] = f1_stasic2[func_id-1][:]
        recordeddata[:,12] = f1_stasic3[func_id-1][:]  
        recordeddata[:,13] = f1_stasic4[func_id-1][:]
        recordeddata[:,14] = f1_stasic5[func_id-1][:]
        datafram_recordeddata = pd.DataFrame(recordeddata)
        recoed_filename = "./recorded_data/fuction" + str(func_id) + ".csv"
        datafram_recordeddata.to_csv(recoed_filename)

    # save data to txt
    import sys
    newfile = 'result.txt'
    data = open(newfile,'a',encoding="utf-8")
    for func_id in range(1, problemcnt):
        strproblem = "+Problem " + str(func_id)
        print(strproblem,file=data)

        strrecall = "recall: " + str(recall1[func_id-1][:]) + " mean: " + str(np.mean(recall1[func_id-1][:])) + " var: " + str(np.var(recall1[func_id-1][:]))
        print(strrecall,file=data)
        strrecall = "recall: " + str(recall2[func_id-1][:]) + " mean: " + str(np.mean(recall2[func_id-1][:])) + " var: " + str(np.var(recall2[func_id-1][:]))
        print(strrecall,file=data)
        strrecall = "recall: " + str(recall3[func_id-1][:]) + " mean: " + str(np.mean(recall3[func_id-1][:])) + " var: " + str(np.var(recall3[func_id-1][:]))
        print(strrecall,file=data)
        strrecall = "recall: " + str(recall4[func_id-1][:]) + " mean: " + str(np.mean(recall4[func_id-1][:])) + " var: " + str(np.var(recall4[func_id-1][:]))
        print(strrecall,file=data)
        strrecall = "recall: " + str(recall5[func_id-1][:]) + " mean: " + str(np.mean(recall5[func_id-1][:])) + " var: " + str(np.var(recall5[func_id-1][:]))
        print(strrecall,file=data)

        runstr = "-------------------"
        print(runstr,file=data) 
        strprecision = "precision: " + str(precison1[func_id-1][:]) + " mean: " + str(np.mean(precison1[func_id-1][:])) + " var: " + str(np.var(precison1[func_id-1][:]))
        print(strprecision,file=data)
        strprecision = "precision: " + str(precison2[func_id-1][:]) + " mean: " + str(np.mean(precison2[func_id-1][:])) + " var: " + str(np.var(precison2[func_id-1][:]))
        print(strprecision,file=data)                
        strprecision = "precision: " + str(precison3[func_id-1][:]) + " mean: " + str(np.mean(precison3[func_id-1][:])) + " var: " + str(np.var(precison3[func_id-1][:]))
        print(strprecision,file=data)        
        strprecision = "precision: " + str(precison4[func_id-1][:]) + " mean: " + str(np.mean(precison4[func_id-1][:])) + " var: " + str(np.var(precison4[func_id-1][:]))
        print(strprecision,file=data)
        strprecision = "precision: " + str(precison5[func_id-1][:]) + " mean: " + str(np.mean(precison5[func_id-1][:])) + " var: " + str(np.var(precison5[func_id-1][:]))
        print(strprecision,file=data)

        runstr = "-------------------"
        print(runstr,file=data) 
        strf1 = "f1_stastic " + str(f1_stasic1[func_id-1][:]) + " mean: " + str(np.mean(f1_stasic1[func_id-1][:])) + " var: " + str(np.var(f1_stasic1[func_id-1][:]))
        print(strf1,file=data)
        strf1 = "f1_stastic " + str(f1_stasic2[func_id-1][:]) + " mean: " + str(np.mean(f1_stasic2[func_id-1][:])) + " var: " + str(np.var(f1_stasic2[func_id-1][:]))
        print(strf1,file=data)        
        strf1 = "f1_stastic " + str(f1_stasic3[func_id-1][:]) + " mean: " + str(np.mean(f1_stasic3[func_id-1][:])) + " var: " + str(np.var(f1_stasic3[func_id-1][:]))
        print(strf1,file=data)
        strf1 = "f1_stastic " + str(f1_stasic4[func_id-1][:]) + " mean: " + str(np.mean(f1_stasic4[func_id-1][:])) + " var: " + str(np.var(f1_stasic4[func_id-1][:]))
        print(strf1,file=data)        
        strf1 = "f1_stastic " + str(f1_stasic5[func_id-1][:]) + " mean: " + str(np.mean(f1_stasic5[func_id-1][:])) + " var: " + str(np.var(f1_stasic5[func_id-1][:]))
        print(strf1,file=data)
        runstr = "\n"
        print(runstr,file=data) 

if __name__ == '__main__':
    experiment()


    # parse = argparse.ArgumentParser()
    # parse.add_argument('--func_id',default=1,type=int)
    # args = parse.parse_args()
    # func_id = args.func_id
    # func_id = 3
    # ans = run(func_id)
    # # see_result(ans[:, :-1], CEC2013(func_id))
    # print_result(ans[:, :-1], CEC2013(func_id),1,2)
    # np.savetxt('points.txt', ans)
