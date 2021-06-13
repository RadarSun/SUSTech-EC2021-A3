# import numpy as np
# from cec2013.cec2013 import *
# from modules import *
# import argparse


# def main(func_id):
#     f = CEC2013(func_id)
#     d = f.get_dimension()
#     pop_size = 16 * d
#     archive = np.empty((0, d + 1))
#     evaluation_remaind_cnt = f.get_maxfes()
#     # evaluation_remaind_cnt *= 10
#     v = 1
#     for i in range(f.get_dimension()):
#         v *= f.get_ubound(i) - f.get_lbound(i)
#     pre_p = np.empty((0, d))
#     radius = np.array([])
#     while evaluation_remaind_cnt > 0:
#         P = reject_init(f, 4 * pop_size, pre_p, radius)
#         evaluation_remaind_cnt -= P.shape[0]
#         score = np.zeros(P.shape[0])
#         for i in range(P.shape[0]):
#             score[i] = f.evaluate(P[i])
#         pos = np.argsort(score)[::-1]
#         P = P[pos[:1 * pop_size]]
#         temp_p = np.empty((0, d))
#         temp_radius = np.array([])
        
#         for p in P:
#             p2, evaluation_remaind_cnt = cmsa2_reject(np.reshape(p, (-1, d)), evaluation_remaind_cnt, f, np.ones(1) * f.evaluate(p),
#                                        pre_p, radius)
#             dist = np.linalg.norm(p2[0] - p, 2)
#             temp_radius = np.hstack((temp_radius, dist))
#             temp_p = np.vstack((temp_p, p2[0]))
#             archive, evaluation_remaind_cnt = union(archive, p2, f, evaluation_remaind_cnt)
#         pre_p = temp_p
#         radius = temp_radius
#         count, seeds = how_many_goptima(archive[:, :-1], CEC2013(func_id), 10 ** -5)
#         if count == f.get_no_goptima():
#             return archive
#     return archive


# def experiment():
#     for exp in range(50):
#         for func_id in range(1, 21):
#             print('START', func_id, 'RUN', exp)
#             # ans = main4(func_id)
#             ans = main(func_id)
#             score = np.zeros(ans.shape[0])
#             f = CEC2013(func_id)
#             for i in range(ans.shape[0]):
#                 score[i] = f.evaluate(ans[i, :-1])
#             ans = np.hstack((ans, score.reshape(ans.shape[0], 1)))
#             np.savetxt('./ans/' + 'problem%03d' % (func_id) + 'run%03d' % (exp+1) + '.dat', ans)


# if __name__ == '__main__':
#     experiment()
import numpy as np
from cec2013.cec2013 import *
from modules2 import *
import argparse


def run(func_id):
    f = CEC2013(func_id)
    D = f.get_dimension()
    pop_size = 16 * D
    archive = np.empty((0, D))
    evaluation_remaind_cnt = f.get_maxfes()

    pre_p = np.empty((0, D))
    radius = np.array([])
    while evaluation_remaind_cnt > 0:
        population = initialization(f, 4 * pop_size, pre_p, radius)
        evaluation_remaind_cnt -= population.shape[0]
        score = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            score[i] = f.evaluate(population[i])
        pos = np.argsort(score)[::-1]
        population = population[pos[:1 * pop_size]]# decrese 
        temp_p = np.empty((0, D))
        temp_radius = np.array([])

        for sub_population in population:
            p2, evaluation_remaind_cnt = cmsa(np.reshape(sub_population, (-1, D)), evaluation_remaind_cnt, f, np.ones(1) * f.evaluate(sub_population),
                                       pre_p, radius)
            dist = np.linalg.norm(p2[0] - sub_population, 2)
            temp_radius = np.hstack((temp_radius, dist))
            temp_p = np.vstack((temp_p, p2[0]))
            archive, evaluation_remaind_cnt = union(archive, p2, f, evaluation_remaind_cnt)
        print('archive shape,', archive.shape)
        print('eva remain,', evaluation_remaind_cnt)
        see_result(archive[:, :-1], f)
        pre_p = temp_p
        radius = temp_radius
    return archive

def experiment():
    for exp in range(50):
        for func_id in range(1, 21):
            print('START', func_id, 'RUN', exp)
            # ans = main4(func_id)
            ans = run(func_id)
            score = np.zeros(ans.shape[0])
            f = CEC2013(func_id)
            for i in range(ans.shape[0]):
                score[i] = f.evaluate(ans[i, :-1])
            ans = np.hstack((ans, score.reshape(ans.shape[0], 1)))
            np.savetxt('./ans/' + 'problem%03d' % (func_id) + 'run%03d' % (exp+1) + '.dat', ans)


if __name__ == '__main__':
    experiment()

# if __name__ == '__main__':
#     parse = argparse.ArgumentParser()
#     parse.add_argument('--func_id',default=1,type=int)
#     args = parse.parse_args()
#     func_id = args.func_id
#     func_id = 8
#     ans = run(func_id)
#     see_result(ans[:, :-1], CEC2013(func_id))
#     np.savetxt('points.txt', ans)
