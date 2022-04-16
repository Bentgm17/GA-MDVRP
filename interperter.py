from pprint import pformat
from numpy import product
import os
from Coordinate import Coordinate
import matplotlib.pyplot as plt
import math
import json
import sys

def read_instance(path):
    # Read instance
    # print(instance_sub_file,instance_num)
    parent_dir= os.path.dirname(os.path.realpath(__file__))+'/Instances'
    instance = open(os.path.join(parent_dir,path), "r")

    DATASET = instance.readline().split()[2]

    instance.readline()

    DAYS = int(instance.readline().split()[2])
    TRUCK_CAPACITY = int(instance.readline().split()[2])
    TRUCK_MAX_DISTANCE = int(instance.readline().split()[2])
    VAN_CAPACITY = int(instance.readline().split()[2])
    VAN_MAX_DISTANCE = int(instance.readline().split()[2])

    instance.readline()

    TRUCK_DISTANCE_COST = int(instance.readline().split()[2])
    TRUCK_DAY_COST = int(instance.readline().split()[2])
    TRUCK_COST = int(instance.readline().split()[2])
    VAN_DISTANCE_COST = int(instance.readline().split()[2])
    VAN_DAY_COST = int(instance.readline().split()[2])
    VAN_COST = int(instance.readline().split()[2])

    instance.readline()

    DELIVER_EARLY_PENALTY = int(instance.readline().split()[2])

    instance.readline()

    NUMBER_OF_PRODUCTS = int(instance.readline().split()[2])

    products = {}

    for i in range(NUMBER_OF_PRODUCTS):
        line = instance.readline().split()
        products[int(line[0])] = int(line[1])

    instance.readline()

    NUMBER_OF_HUBS = int(instance.readline().split()[2])

    hubs = {}

    for i in range(NUMBER_OF_HUBS):
        line = instance.readline().split()
        hubs[int(line[0])] = {
            'location': i + 2,

            'cost': int(line[1]),
            
            # Possible for Delivery
            'PFD': [int(i) for i in line[2].split(',')]
        }

    # print(hubs)


    instance.readline()

    NUMBER_OF_LOCATIONS = int(instance.readline().split()[2])
    depot_line = instance.readline().split()
    DEPOT_COORDINATE=Coordinate(int(depot_line[1]), int(depot_line[2]))


    for i in (range(NUMBER_OF_HUBS)):
        line = instance.readline().split()
        hubs[i+1]['location'] = Coordinate(int(line[1]), int(line[2]))
    locations = {}

    for i in range(NUMBER_OF_LOCATIONS-NUMBER_OF_HUBS-1):
        line = instance.readline().split()
        locations[int(line[0])] = Coordinate(int(line[1]), int(line[2]))
    instance.readline()

    dict_to_json={'depot_xy':[[]],'customer_xy':[[]]}

    for hub in hubs.values():
        dict_to_json['depot_xy'][0].append([hub['location'].get_x(),hub['location'].get_y()])


    NUMBER_OF_REQUESTS = int(instance.readline().split()[2])

    requests = {}

    for i in range(NUMBER_OF_REQUESTS):
        line = instance.readline().split()

        requests[int(line[0])] = {
            'request_day': int(line[1]),
            'location': locations[int(line[2])],
            'num_of_products': {key: int(value) for (key, value) in enumerate(line[3].split(','))}
        }

    request_day=set([day['request_day'] for day in requests.values()])
    request_products=[day['num_of_products'] for day in requests.values()]
    print(request_products)
    sys.exit()
    for key,value in requests[int(line[0])]['num_of_products'].items():
        for location in locations.values():
            dict_to_json['customer_xy'][0].append([location.get_x(),location.get_y()])
            print(dict_to_json)


    instance.close()

    depot_x=DEPOT_COORDINATE.get_x()
    depot_y=DEPOT_COORDINATE.get_y()

    hub_x=[]
    hub_y=[]

    for _,value in hubs.items():
        hub_x.append(value['location'].get_x())
        hub_y.append(value['location'].get_y())

for folder in os.listdir('Instances'):
    if os.path.isdir('Instances/'+folder): 
        for file in os.listdir('Instances/'+folder):
            read_instance(folder+"/"+file)
    #  callthecommandhere(blablahbla, filename, foo)