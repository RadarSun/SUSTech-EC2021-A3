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

        for individual in population:
            new_population, evaluation_remaind_cnt = cmsa(np.reshape(individual, (-1, D)), evaluation_remaind_cnt, f, np.ones(1) * f.evaluate(individual),
                                       taboo_points, radius_of_taboo_points)
            radius_of_this_taboo = np.linalg.norm(new_population[0] - individual, 2)
            tmp_radius_of_taboo_points = np.hstack((tmp_radius_of_taboo_points, radius_of_this_taboo))
            tmp_taboo_point = np.vstack((tmp_taboo_point, new_population[0]))
            archive, evaluation_remaind_cnt = union(archive, new_population, f, evaluation_remaind_cnt)
        # print('archive shape,', archive.shape)
        # print('eva remain,', evaluation_remaind_cnt)
        # see_result(archive[:, :-1], f) 
        taboo_points = tmp_taboo_point
        radius_of_taboo_points = tmp_radius_of_taboo_points
    return archive


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--func_id',default=1,type=int)
    args = parse.parse_args()
    func_id = args.func_id
    func_id = 3
    ans = run(func_id)
    # see_result(ans[:, :-1], CEC2013(func_id))
    print_result(ans[:, :-1], CEC2013(func_id),1,2)
    np.savetxt('points.txt', ans)
