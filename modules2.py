import numpy as np
from cec2013.cec2013 import *
import collections
import time

def hv_test(individual1, individual2, f, N=5):
    '''
    judge whether two individuals are close, if so, return True
    the second return para is evaluatim_cnt
    '''
    are_they_close = True
    evaluatim_cnt = 0
    if (np.linalg.norm(individual1 - individual2, 2) >= 1e-7):
        v = min(f.evaluate(individual1), f.evaluate(individual2))
        data = np.linspace(individual1, individual2, N + 2)
        for i in range(1, N + 1):
            if f.evaluate(data[i]) < v:
                are_they_close = False
                evaluatim_cnt = i
    return are_they_close, evaluatim_cnt

def nearest_hv_test(individual, P, f):
    '''
    judge whether the individual is in the archive, if so, return True 
    the second return para is evaluatim_cnt
    '''
    if P.shape[0] == 0:
        return False, 0
    dist = np.linalg.norm(P - individual, axis=1) # get the 2 norm of each row vector of a matrix
    idx = np.argmin(dist) # find the nearest one
    return hv_test(individual, P[idx], f)


def union(archive, new_population, f, evaluation_remaind_cnt,start_time):
    '''
    update the archive and evaluation_remianed_cnt
    use new population which obtained by CMSA-EA
    archive includes the optimal solutions
    '''
    eva_cost = 0
    if new_population.shape[0] == 0:
        return archive

    # the new population is sorted in descending order    
    score = np.zeros(new_population.shape[0])
    for i in range(new_population.shape[0]):
        score[i] = f.evaluate(new_population[i])
    pos = np.argsort(score)[::-1]
    new_population = new_population[pos]

     # first time, add the new population directly, last colums is the rest time of total evaluation 
    nowtime = time.time()*1000
    nowtime = nowtime - start_time
    if archive.shape[0] == 0:
        tmp_archive = np.empty((0, new_population.shape[1]))
        for i in range(new_population.shape[0]):
            flag, eva = nearest_hv_test(new_population[i], tmp_archive, f)
            eva_cost += eva
            if not flag:
                tmp_archive = np.vstack((tmp_archive, new_population[i]))
        archive = np.hstack( (tmp_archive, np.reshape(np.ones(tmp_archive.shape[0]) * evaluation_remaind_cnt, (-1, 1)), \
            np.reshape(np.ones(tmp_archive.shape[0]) * nowtime, (-1, 1))) )
        evaluation_remaind_cnt -= eva_cost
   
    # if not the first time
    else:
        # update the max score by archive and new population
        max_score = np.max(score)
        for i in range(archive.shape[0]):
            max_score = max(f.evaluate(archive[i, :-2]), max_score)

        # update the archive by archive around the max score 
        tmp_archive = np.empty((0, f.get_dimension() + 2))
        for i in range(archive.shape[0]):
            if abs(f.evaluate(archive[i, :-2]) - max_score) < 1e-1:
                tmp_archive = np.vstack((tmp_archive, archive[i]))
        archive = tmp_archive

        # update the archive by new population around archive
        for i in range(new_population.shape[0]):
            if abs(score[pos[i]] - max_score) < 1e-1:
                flag, eva = nearest_hv_test(new_population[i], archive[:, :-2], f)
                eva_cost += eva
                if not flag: # if not True
                    archive = np.vstack((archive, np.hstack((new_population[i], evaluation_remaind_cnt,nowtime))))
        evaluation_remaind_cnt -= eva_cost
    return archive,evaluation_remaind_cnt

def initialization(f, pop_size, archive, radius):
    D = f.get_dimension()
    polulation = np.zeros((pop_size, D))
    ub = np.zeros(D)
    lb = np.zeros(D)
    # Get lower, upper bounds
    for k in range(D):
        ub[k] = f.get_ubound(k)
        lb[k] = f.get_lbound(k)
    # Create population within bounds
    for j in range(pop_size):
        polulation[j] = lb + (ub - lb) * np.random.rand(1, D)
        flag = True
        for k in range(archive.shape[0]):
            if np.linalg.norm(archive[k] - polulation[j], 2) < radius[k]:
                flag = False
                break
        while flag == False and np.random.rand() < 0.9:
            # print(flag)
            polulation[j] = lb + (ub - lb) * np.random.rand(1, D)
            flag = True
            for k in range(archive.shape[0]):
                if np.linalg.norm(archive[k] - polulation[j], 2) < radius[k]:
                    flag = False
                    break
    return polulation



def Sample(mean, cov, lam, f, sigma):
    ans = np.array([])
    tau = 1 / np.sqrt(2 * f.get_dimension())
    d = f.get_dimension()
    new_sigma = np.zeros(lam)
    ub = np.zeros(d)
    lb = np.zeros(d)
    for i in range(d):
        ub[i] = f.get_ubound(i)
        lb[i] = f.get_lbound(i)

    for i in range(lam):
        x = np.random.multivariate_normal(np.zeros(d), cov)
        step = np.exp(tau * np.random.normal())
        new_sigma[i] = step * sigma
        x = mean + x * new_sigma[i]
        for j in range(d):
            if x[j] > ub[j]:
                x[j] = ub[j]
            if x[j] < lb[j]:
                x[j] = lb[j]
        if ans.shape[0] == 0:
            ans = np.reshape(x, (-1, d))
        else:
            ans = np.vstack((ans, x))
    return ans, new_sigma


def see_result(Archive, f):
    score = np.zeros(Archive.shape[0])
    for i in range(Archive.shape[0]):
        score[i] = f.evaluate(Archive[i])
    print('infor', f.get_info())
    print('best so far', np.max(score))
    for err in range(-1, -6, -1):
        count, seeds = how_many_goptima(Archive, f, 10 ** err)
        print("Let err be ", 10 ** err, " In the current population there exist", count, "global optimizers.")

def print_result(Archive,f,f_id,run_id):
    score = np.zeros(Archive.shape[0])
    for i in range(Archive.shape[0]):
        score[i] = f.evaluate(Archive[i])
    # print('infor', f.get_info())
    # print('best so far', np.max(score))
    dist = f.get_info()
    num_optimal = dist["nogoptima"]
    import sys
    newfile = 'test.txt'
    data = open(newfile,'a',encoding="utf-8")
    runstr = "\n-----------------------\n Problem " + str(f_id) + " Run " + str(run_id)
    print(runstr,file=data)


    recall = np.zeros(5)
    precision = np.zeros(5)
    f1_stasic = np.zeros(5)
    for err in range(-1, -6, -1):
        count, seeds = how_many_goptima(Archive, f, 10 ** err)
        Recall = float(count)/float(num_optimal)
        if Recall > 1:
            Recall = 1
        recall[-err-1] = Recall
        Precision = float(count)/float(Archive.shape[0])
        if Precision > 1:
            Precision = 1
        precision[-err-1] = Precision
        F1_stastic = Recall*Precision
        f1_stasic[-err-1] = F1_stastic

        str1 = "Err: " + str(10 ** err) + "  Global optimizers: " + str(num_optimal) + "  Find: " + str(count) 
        str2 = "Recall: " + '{:.7f}'.format(Recall) + "  Precision: " + '{:.7f}'.format(Precision) + "  F1_stastic: " + '{:.7f}'.format(F1_stastic) 
        # print("Err: ", 10 ** err, "  Global optimizers: ",num_optimal, "  Find: ", count)
        # print("Recall: ",'{:.7f}'.format(Recall), "Precision: ",'{:.7f}'.format(Precision), "F1_stastic: ",'{:.7f}'.format(F1_stastic))
        data = open(newfile,'a',encoding="utf-8")
        print(str1,file=data)
        data = open(newfile,'a',encoding="utf-8")
        print(str2,file=data)
        data.close()
    return recall,precision,f1_stasic


def reject(best_individual, archive, radius):
    if archive.shape[0] == 0:
        is_in_reject_region = False
        return is_in_reject_region
    '''
    if the best individual is in the reject region of 
    the nearest reject point, reject it
    '''    
    dist = np.linalg.norm(archive - best_individual, 2, axis=1)
    idx = np.argmin(dist)
    if dist[idx] < radius[idx]:
        is_in_reject_region = True
    else:
        is_in_reject_region = False
    return is_in_reject_region

def cmsa(subpopulation, evaluation_remaind_cnt, f, score_of_subpop, archive, radius):
    D = f.get_dimension()
    x_mean = np.mean(subpopulation, axis=0)
    sigma = np.ones(subpopulation.shape[0])
    sigma_mean = np.mean(sigma)
    cov = np.eye(D)
    
    lamda_subpop_size = int((4.5 + 3 * np.log(D)))
    lam = lamda_subpop_size
    mu = int(lam/2)
    elite_num = max(1, int(0.15 * mu))
    MAX_DEPTH = 10 + int(30 * D / mu)
    depth = 0
    best_score = f.evaluate(subpopulation[0])
    cnt_double_lam = 0
    while depth < MAX_DEPTH and evaluation_remaind_cnt > 0:
        if (depth != 0 and depth % 5 == 0) and reject(subpopulation[0], archive, radius):
            return subpopulation, evaluation_remaind_cnt
        '''
        combine elite_num old individuals and lam new individuals,
        and then sort the total subpopulation in descending order
        '''
        # define the num of new individuals, doubled
        # lam = min(2 * lamda_subpop_size, evaluation_remaind_cnt) 
        if cnt_double_lam<1:
            lam = min(2 * lam, evaluation_remaind_cnt) 
            mu = int(lam/2)
            elite_num = max(1, int(0.15 * mu))
            cnt_double_lam = cnt_double_lam + 1 

        # produce lam new individuals
        new_population, new_sigma = Sample(x_mean, cov, lam, f, sigma_mean)
    
        # evaluate new individuals
        new_score_of_pop = np.zeros(lam)
        for i in range(lam): 
            new_score_of_pop[i] = f.evaluate(new_population[i])
        evaluation_remaind_cnt -= lam

        # obtain total pop by combining pre elite individuals with new individuals
        total_subpop = np.vstack((subpopulation[:elite_num], new_population))
        total_sigma = np.hstack((sigma[:elite_num], new_sigma))
        total_score_of_subpop = np.hstack((score_of_subpop[:elite_num], new_score_of_pop))

        # sort the total pop
        id_sorted_totalpop = np.argsort(total_score_of_subpop)[::-1]
        subpopulation = total_subpop[id_sorted_totalpop]
        if evaluation_remaind_cnt == 0:
            break
        sigma = total_sigma[id_sorted_totalpop]
        score_of_subpop = total_score_of_subpop[id_sorted_totalpop]

        if score_of_subpop[0] > best_score + 1e-6:
            depth = 0
            best_score = score_of_subpop[0]
        else:
            depth += 1
        
        '''
        use the top mu individual of total subpop to update parameters
        ''' 
        # upadate weight w
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 1) - np.log(i + 1)
        w = w / np.sum(w)

        # update Ci cov
        tau_c = 1 + D * (D + 1) / (mu)
        x_mean = np.mean(subpopulation[:mu], axis=0)
        z = np.zeros_like(subpopulation[:mu])
        for i in range(mu):
            z[i,:] = (subpopulation[i] - x_mean) / sigma[i]
        cov *= (1 - 1 / tau_c)
        for i in range(mu):
            cov = cov + 1 / tau_c * (w[i] * np.matmul(np.reshape(z[i], (-1, 1)), np.reshape(z[i], (1, -1))))
        # the subpopulation should be terminated if the updated Ci has a negative eigenvalue
        if np.min(np.linalg.eigh(cov)[0]) < 0:
            return subpopulation, evaluation_remaind_cnt

        # update x_mean 
        x_mean = np.zeros_like(subpopulation[0])
        for i in range(mu):
            x_mean = x_mean + w[i] * subpopulation[i]

        # update the sigma_mean
        numerator = 1
        denominator = 1
        lam_plus_nelite = lam + elite_num
        for i in range(mu):
            numerator *= sigma[i] ** (w[i])
        for i in range(lam_plus_nelite):
            denominator *= sigma[i] ** (1.0 / lam_plus_nelite)
        sigma = np.ones(mu) * sigma_mean * numerator / denominator
        sigma_mean = np.mean(sigma)

    return subpopulation, evaluation_remaind_cnt

