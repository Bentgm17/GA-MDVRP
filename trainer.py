import random
from copy import deepcopy
import math
import re
import matplotlib.pyplot as plt
import numpy as np
from time import time
import sys
from time import time
import trainer

from requests import request

from Classes import Depot, Customer, Hub

from interperter import Instance

depots = None
customers = None
population = None

class GeneticAlgorithm:

    def __init__(self,instance):
        self.instance=instance
        self.population_size = 2000   
        self.cost=0
        

    def distance(self,pos1, pos2):
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)


    def load_problem(self):
        global hubs, requests
        hubs = self.instance.hubs
        requests = self.instance.requests


    def find_closest_depot(self,request):
        closest_depot = None
        closest_distance = -1
        for i, hub in enumerate(hubs):
            if request in hub.possible_requests:
                d = self.distance(hub.loc, request.customer.loc)
                if closest_depot is None or d < closest_distance:
                    closest_depot = (hub, i)
                    closest_distance = d
        return closest_depot[0], closest_depot[1], closest_distance


    def is_consistent_route(self,route, hub, include_reason=False,indicate=False):
        # print('route:', route)
        route_load = 0
        route_duration = 0
        last_pos = hub.loc
        # same_day=all([requests[rid - 1].day==requests[0].day for rid in route])
        if route:
            earliest_day=min([requests[rid - 1].day for rid in route])
            good_delivery=True
            for r in route:
                request = requests[r - 1]
                if request.day-earliest_day>self.instance.MIN_DAYS_FRESH:
                    good_delivery=False
                route_load += sum(request.demand.values())
                route_duration += self.distance(last_pos, request.customer.loc)
                last_pos = request.customer.loc
            route_duration += self.distance(last_pos, hub.loc)

            if include_reason:
                if route_load > self.instance.VAN_CAPACITY:
                    return False, 1
                if route_duration > self.instance.VAN_MAX_DISTANCE:
                    return False, 2
                return True, 0
            return route_load <= self.instance.VAN_CAPACITY and (self.instance.VAN_MAX_DISTANCE == 0 or route_duration <= self.instance.VAN_MAX_DISTANCE) and good_delivery
        return False

 
    def is_consistent(self,chromosome,indicate=False):
        for r in requests:
            if r.id not in chromosome:
                return False

        routes = self.decode(chromosome)
        for h in range(len(routes)):
            hub = hubs[h]
            for route in routes[h]:
                if not self.is_consistent_route(route, hub,True):
                    return False
        return True


    def encode(self,routes):
        chromosome = []
        for d in range(len(routes)):
            if d != 0:
                chromosome.append(-1)
            for r in range(len(routes[d])):
                if r != 0:
                    chromosome.append(0)
                chromosome.extend(routes[d][r])
        return chromosome


    def decode(self,chromosome):
        routes = [[[]]]
        d = 0
        r = 0
        for i in chromosome:
            if i < 0:
                routes.append([[]])
                d += 1
                r = 0
            elif i == 0:
                routes[d].append([])
                r += 1
            else:
                routes[d][r].append(i)
        return routes

    # calculate and return fitness
    def evaluate(self,chromosome, return_distance=False,return_cost=False):
        day_dict={i:0 for i in range(1,self.instance.PERIOD+1)}
        for r in requests:
            if r.id not in chromosome:
                if return_distance:
                    return math.inf
                return 0

        routes = self.decode(chromosome)
        score = 0
        cost=0
        for hub_index in range(len(routes)):
            hub = hubs[hub_index]
            if len(routes[hub_index])>0:
                days_used=[min([requests[rid - 1].day for rid in route] for route in routes[hub_index])][0]
                cost+=hub.cost*len(days_used)
                score+=hub.cost*len(days_used) 
                [day_dict.__setitem__(day, day_dict[day] + 1) for day in days_used] 
            for route in routes[hub_index]:
                # print([requests[rid - 1].day for rid in route])
                if route:
                    cost+=self.instance.VAN_DAY_COST
                    earliest_day=min([requests[rid - 1].day for rid in route])
                    # if self.instance.DELIVER_EARLY_PENALTY==0:
                    score +=(sum([requests[x - 1].day-earliest_day for x in route])!=0)*5000
                else:
                    earliest_day=0

                route_length, route_load,freshness_cost = self.evaluate_route(route, hub,earliest_day, True)

                score+= route_length*self.instance.VAN_DISTANCE_COST+freshness_cost
                cost+=route_length*self.instance.VAN_DISTANCE_COST+freshness_cost

                if route_length > self.instance.VAN_MAX_DISTANCE:
                    score += (route_length - self.instance.VAN_MAX_DISTANCE) * 50
                if route_load > self.instance.VAN_CAPACITY:
                    score += (route_load - self.instance.VAN_MAX_DISTANCE) * 50
        score+=max(day_dict.values())*self.instance.VAN_COST
        cost+=max(day_dict.values())*self.instance.VAN_COST
        
        if return_cost:
            return cost
        if return_distance:
            return score
        return 1/score

    def compute_fresh_cost(self,request, earliest_day):
        return self.instance.DELIVER_EARLY_PENALTY**(request.day-earliest_day)


    def evaluate_route(self,route, hub,earliest_day, return_load=False):
        if len(route) == 0:
            if return_load:
                return 0, 0, 0
            return 0
        route_load = 0
        route_length = 0
        freshness_cost=0
        request = None
        last_pos = hub.loc
        for rid in route:
            request = requests[rid - 1]
            if self.instance.DELIVER_EARLY_PENALTY==0:
                freshness_cost += (request.day-earliest_day)*10
            else:
                freshness_cost += self.compute_fresh_cost(request,earliest_day)
            route_load += sum(request.demand.values())
            route_length += self.distance(last_pos, request.customer.loc)
            last_pos = request.customer.loc
        route_length += self.distance(last_pos, hub.loc)

        if return_load:
            return route_length, route_load,freshness_cost
        return route_length


    def schedule_route(self,route):
        if not len(route):
            return route
        new_route = []
        prev_cust = random.choice(route)
        route.remove(prev_cust)
        new_route.append(prev_cust)

        while len(route):
            prev_cust = min(route, key=lambda x: self.distance(requests[x - 1].customer.loc, requests[prev_cust - 1].customer.loc))
            route.remove(prev_cust)
            new_route.append(prev_cust)
        return new_route


    def create_heuristic_chromosome(self,groups):
        # Group customers in routes according to savings
        routes = [[] for i in range(len(hubs))]
        missing_requests = list(map(lambda x: x.id, requests))
        for h in range(len(groups)):
            hub = hubs[h]
            savings = []
            for i in range(len(groups[h])):
                ci = requests[groups[h][i] - 1]
                savings.append([])
                for j in range(len(groups[h])):
                    if j <= i:
                        savings[i].append(0)
                    else:
                        cj = requests[groups[h][j] - 1]
                        savings[i].append(self.distance(hub.loc, ci.customer.loc) + self.distance(hub.loc, cj.customer.loc) -
                                        self.distance(ci.customer.loc, cj.customer.loc))
            savings = np.array(savings)
            order = np.flip(np.argsort(savings, axis=None), 0)

            for saving in order:
                i = saving // len(groups[h])
                j = saving % len(groups[h])

                ci = groups[h][i]
                cj = groups[h][j]

                ri = -1
                rj = -1
                for r, route in enumerate(routes[h]):
                    if ci in route:
                        ri = r
                    if cj in route:
                        rj = r

                route = None
                if ri == -1 and rj == -1:
                        route = [ci, cj]
                elif ri != -1 and rj == -1:
                    if routes[h][ri].index(ci) in (0, len(routes[h][ri]) - 1):
                        route = routes[h][ri] + [cj]
                elif ri == -1 and rj != -1:
                    if routes[h][rj].index(cj) in (0, len(routes[h][rj]) - 1):
                        route = routes[h][rj] + [ci]
                elif ri != rj:
                    route = routes[h][ri] + routes[h][rj]

                if route:
                    if self.is_consistent_route(route, hub, include_reason=True)[1] == 2:
                        route = self.schedule_route(route)
                    if self.is_consistent_route(route, hub):
                        if ri == -1 and rj == -1:
                            routes[h].append(route)
                            missing_requests.remove(ci)
                            if ci != cj:
                                missing_requests.remove(cj)
                        elif ri != -1 and rj == -1:
                            routes[h][ri] = route
                            missing_requests.remove(cj)
                        elif ri == -1 and rj != -1:
                            routes[h][rj] = route
                            missing_requests.remove(ci)
                        elif ri != -1 and rj != -1:
                            if ri > rj:
                                routes[h].pop(ri)
                                routes[h].pop(rj)
                            else:
                                routes[h].pop(rj)
                                routes[h].pop(ri)
                            routes[h].append(route)


        # Order customers within routes
        for i, hub_routes in enumerate(routes):
            for j, route in enumerate(hub_routes):
                new_route = self.schedule_route(route)
                routes[i][j] = new_route

        chromosome = self.encode(routes)
        chromosome.extend(missing_requests)
        return chromosome


    def create_random_chromosome(self,groups):
        routes = []
        for h in range(len(groups)):
            hub = hubs[h]
            group = groups[h][:]
            random.shuffle(group)
            routes.append([[]])

            r = 0
            route_cost = 0
            route_load = 0
            last_pos = hub.loc
            for c in requests:
                request = requests[c - 1]
                cost = self.distance(last_pos, request.customer.loc) 
                if route_cost + cost > self.instance.VAN_CAPACITY or route_load + sum(request.demand.values())> self.instance.VAN_MAX_DISTANCE:
                    r += 1
                    routes[h].append([])
                routes[h][r].append(c)

        return self.encode(routes)


    def initialize(self,random_portion=0):
        global population
        population = []
        groups = [[] for i in range(len(hubs))]

        # Group customers to closest depot
        for r in requests:
            hub, hub_index, dist = self.find_closest_depot(r)
            groups[hub_index].append(r.id)

        for z in range(int(self.population_size * (1 - random_portion))):
            chromosome = self.create_heuristic_chromosome(groups)
            population.append((chromosome, self.evaluate(chromosome)))

        for z in range(int(self.population_size * random_portion)):
            chromosome = self.create_random_chromosome(groups)
            population.append((chromosome, self.evaluate(chromosome)))


    def select(self,portion, elitism=0):
        total_fitness = sum(map(lambda x: x[1], population))
        weights = list(map(lambda x: (total_fitness - x[1])/(total_fitness * (self.population_size - 1)), population))
        selection = random.choices(population, weights=weights, k=int(self.population_size*portion - elitism))
        population.sort(key=lambda x: -x[1])
        if elitism > 0:
            selection.extend(population[:elitism])
        return selection


    def crossover(self,p1, p2):
        protochild = [None] * max(len(p1), len(p2))
        cut1 = int(random.random() * len(p1))
        cut2 = int(cut1 + random.random() * (len(p1) - cut1))
        substring = p1[cut1:cut2]

        for i in range(cut1, cut2):
            protochild[i] = p1[i]

        p2_ = list(reversed(p2))
        for g in substring:
            if g in p2_:
                p2_.remove(g)
        p2_.reverse()

        j = 0
        for i in range(len(protochild)):
            if protochild[i] is None:
                if j >= len(p2_):
                    break
                protochild[i] = p2_[j]
                j += 1

        i = len(protochild) - 1
        while protochild[i] is None:
            protochild.pop()
            i -= 1

        population.append((protochild, self.evaluate(protochild)))


    def heuristic_mutate(self,p):
        g = []
        for i in range(3):
            g.append(int(random.random() * len(p)))

        offspring = []
        for i in range(len(g)):
            for j in range(len(g)):
                if g == j:
                    continue
                o = p[:]
                o[g[i]], o[g[j]] = o[g[j]], o[g[i]]
                offspring.append((o, self.evaluate(o)))

        selected_offspring = max(offspring, key=lambda o: o[1])
        population.append(selected_offspring)



    def inversion_mutate(self,p):

        cut1 = int(random.random() * len(p))
       
        cut2 = int(cut1 + random.random() * (len(p) - cut1))

        if cut1 == cut2:
            return
        if cut1 == 0:
            child = p[:cut1] + p[cut2 - 1::-1] + p[cut2:]
        else:
            child = p[:cut1] + p[cut2 - 1:cut1 - 1:-1] + p[cut2:]
        population.append((child, self.evaluate(child)))



    def best_insertion_mutate(self,p):
        g = int(random.random() * len(p))

        best_child = None
        best_score = -1000

        for i in range(len(p) - 1):
            child = p[:]
            gene = child.pop(g)
            child.insert(i, gene)
            score = self.evaluate(child)
            if score > best_score:
                best_score = score
                best_child = child

        population.append((best_child, best_score))

    def hub_move_mutate(self,p):
        if -1 not in p:
            return
        i = int(random.random() * len(p))
        while p[i] != -1:
            i = (i + 1) % len(p)

        move_len = int(random.random() * 10) - 5
        new_pos = (i + move_len) % len(p)

        child = p[:]
        child.pop(i)
        child.insert(new_pos, -1)
        population.append((child, self.evaluate(child)))


    def route_merge(self,p):
        routes = self.decode(p)

        d1 = int(random.random() * len(routes))
        r1 = int(random.random() * len(routes[d1]))
        d2 = int(random.random() * len(routes))
        r2 = int(random.random() * len(routes[d2]))

        if random.random() < 0.5:
            limit = int(random.random() * len(routes[d2][r2]))
        else:
            limit = len(routes[d2][r2])

        reverse = random.random() < 0.5

        for i in range(limit):
            if reverse:
                routes[d1][r1].append(routes[d2][r2].pop(0))
            else:
                routes[d1][r1].append(routes[d2][r2].pop())
        routes[d1][r1] = self.schedule_route(routes[d1][r1])
        routes[d2][r2] = self.schedule_route(routes[d2][r2])
        child = self.encode(routes)
        population.append((child, self.evaluate(child)))


    def train(self,generations, crossover_rate, heuristic_mutate_rate, inversion_mutate_rate,
            depot_move_mutate_rate, best_insertion_mutate_rate, route_merge_rate, t1,
            intermediate_plots=True, write_csv=None, log=True):
        global population
        for g in range(generations):
            if log and g % 10 == 0:
                best = max(population, key=lambda x: x[1])
                print(f'[Generation {g}] Best score: {best[1]} Score: {round(self.evaluate(best[0],return_cost=True),2)} Consistent: {self.is_consistent(best[0],indicate=True)}')

            # plottime
            if intermediate_plots and g % 100 == 0:
                if g != 0:
                    best = max(population, key=lambda x: x[1])
                    # population.sort(key=lambda x: -x[1])
                    # self.save_solution(population[0][0],'/Users/gijs/Documents/Business_Analytics/Combinatorial Optimization/Business Case/Genetic Solution/GA/solution.txt')
                    # self.plot(population[0][0])
                    self.save_solution(best[0],'/Users/gijs/Documents/Business_Analytics/Combinatorial Optimization/Business Case/Genetic Solution/GA/solution.txt')
                    self.plot(best[0])

            selection = self.select(heuristic_mutate_rate + inversion_mutate_rate
                            + crossover_rate + depot_move_mutate_rate + best_insertion_mutate_rate
                            + route_merge_rate)
            selection = list(map(lambda x: x[0], selection))

            offset = 0
            for i in range(int((self.population_size * crossover_rate) / 2)):
                p1, p2 = selection[2*i + offset], selection[2*i + 1 + offset]
                self.crossover(p1, p2)
                self.crossover(p2, p1)
            offset += int(self.population_size * crossover_rate)


            for i in range(int(self.population_size * heuristic_mutate_rate)):
                self.heuristic_mutate(selection[i + offset])
            offset += int(self.population_size * heuristic_mutate_rate)

            for i in range(int(self.population_size * inversion_mutate_rate)):
                self.inversion_mutate(selection[i + offset])
            offset += int(self.population_size * inversion_mutate_rate)

            for i in range(int(self.population_size * depot_move_mutate_rate)):
                self.depot_move_mutate(selection[i + offset])
            offset += int(self.population_size * depot_move_mutate_rate)

            for i in range(int(self.population_size * best_insertion_mutate_rate)):
                self.best_insertion_mutate(selection[i + offset])
            offset += int(self.population_size * best_insertion_mutate_rate)

            for i in range(int(self.population_size * route_merge_rate)):
                self.route_merge(selection[i + offset])
            offset += int(self.population_size * route_merge_rate)

            population = self.select(1.0, elitism=math.floor(0.01*self.population_size))
            population=[pop for pop in population if self.is_consistent(pop[0])]

        population.sort(key=lambda x: -x[1])
        print("\n\nFinished training")

        best_score, best_solution = None, None
        if self.is_consistent(population[0][0]):
            best_solution = population[0][0]
            best_score = population[0][1]
            print(f'Best score: {best_score}, best distance: {self.evaluate(best_solution, True)}')
        else:
            for c in population:
                if self.is_consistent(c[0]): 
                    best_solution = c[0]
                    best_score = c[1]
                    print(f'Best score: {best_score}, best distance: {self.evaluate(best_solution, True)}')
                    break
            else:
                print('Found no consistent solutions.')
        print(f'inference time: {time()-t1}s')
        if best_solution:
            if write_csv is not None:
                with open(write_csv, 'a') as f:
                    f.write(f'{time()-t1},{self.evaluate(best_solution, True)/1e2}\n')
            else:
                self.plot(best_solution)
        return best_solution


    def plot_map(self,show=True, annotate=True):
        hub_positions = np.array(list(map(lambda x: x.loc.get_cor(), hubs)))
        requests_positions = np.array(list(map(lambda x: x.customer.loc.get_cor(), requests)))

        hub_ids = np.arange(1, len(hubs) + 1)
        request_id = np.arange(1, len(requests) + 1)

        plt.scatter(hub_positions[:, 0], hub_positions[:, 1], c='r', s=60, zorder=10)
        plt.scatter(requests_positions[:, 0], requests_positions[:, 1], c='k', s=20, zorder=20)

        if annotate:
            for i, id in enumerate(hub_ids):
                plt.annotate(id, hub_positions[i], zorder=30)
            for i, id in enumerate(request_id):
                plt.annotate(id, requests_positions[i], zorder=30)

        if show:
            plt.show()
           


    def plot(self,chromosome):
        r = self.decode(chromosome)
        print('depot No., visit route')
        for h, routes in enumerate(r):
            hub = hubs[h]
            for route in routes:
                positions = [hub.loc.get_cor()]
                last_pos = hub.loc.get_cor()
                for cid in route:
                    last_pos = requests[cid - 1].customer.loc.get_cor()
                    positions.append(last_pos)
                positions.append(hub.loc.get_cor())

                positions = np.array(positions)
                

                plt.plot(positions[:, 0], positions[:, 1], zorder=0)
                print(h+1, route)

        self.plot_map(False)
        plt.savefig('/Users/gijs/Documents/Business_Analytics/Combinatorial Optimization/Business Case/Genetic Solution/GA/solution.png')
        # plt.show()


    def save_solution(self,chromosome, path):
        routes = self.decode(chromosome)
        total_duration = self.evaluate(chromosome, True)

        with open(path, 'w') as f:
            f.write(f'{total_duration:.2f}\n')

            for h, hub in enumerate(hubs):
                for r, route in enumerate(routes[h]):
                    if route:
                        earliest_day=min([requests[rid - 1].day for rid in route])
                        route_length, route_load, freshness_cost = self.evaluate_route(route, hub, earliest_day, True)
                        f.write(f'{h + 1}\t{r + 1}\t{route_length:.2f}\t{route_load}\t{freshness_cost:.2f}\t')
                        end_depot = self.find_closest_depot(requests[route[-1] - 1])[1]
                        f.write(f'{end_depot + 1}\t')

                        f.write(' '.join([str(c) for c in route])+'\t\t\t')
                        f.write(' '.join([str(requests[c-1].day) for c in route])+'\t')
                        f.write('\n')

if __name__=="__main__":
    generations = 1000
    crossover_rate = 0.4
    heuristic_mutate_rate = 0.05
    inversion_mutate_rate = 0.05
    depot_move_mutate_rate = 0
    best_insertion_mutate_rate = 0.05
    route_merge_rate = 0.05
    t1 = time()
    new_file='Instance_11-20/Instance_14.txt'
    new_instance=Instance(new_file)
    new_instance.read_line()
    trainer=GeneticAlgorithm(new_instance)
    trainer.load_problem()  
    trainer.initialize()
    best_solution = trainer.train(generations, crossover_rate, heuristic_mutate_rate, inversion_mutate_rate,
                                depot_move_mutate_rate, best_insertion_mutate_rate, route_merge_rate, t1,
                                intermediate_plots=True, write_csv = sys.argv[0])
    # if best_solution:
    #         trainer.save_solution(best_solution,'/Users/gijs/Documents/Business_Analytics/Combinatorial Optimization/Business Case/Genetic Solution/GA/solution.txt')


