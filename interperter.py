from pprint import pformat
from numpy import product
import os
from Classes import Coordinate,Request,Customer,Hub,Product,Depot
import matplotlib.pyplot as plt
import math
import json
import sys


class Instance():

    def __init__(self,filename):
        parent_dir=os.path.dirname(os.path.realpath(__file__))+'/Instances'
        self.instance=open(os.path.join(parent_dir,filename), "r")

    def capitalized(self,var):
        return var.isupper()
    
    def read_fixed_vars(self,line):
        try:
            exec(f'self.__class__.{line[0]} = {line[2]}')
        except:
            return None

    def read_hubs(self):
        self.depot=None
        self.hubs=[]
        new_line=self.instance.readline()
        while new_line.strip():
            splitted_new_line=new_line.split()
            self.hubs.append(Hub(int(splitted_new_line[0]),int(splitted_new_line[1]),[int(i) for i in splitted_new_line[2].split(',')]))
            new_line=self.instance.readline()
    
    def read_locations(self):
        self.locations=[]
        new_line=self.instance.readline()
        while new_line.strip():
            splitted_new_line=new_line.split()
            self.locations.append(Coordinate(int(splitted_new_line[0]),int(splitted_new_line[1]),int(splitted_new_line[2])))
            new_line=self.instance.readline()
        self.depot=Depot(1,self.locations[0])
        for i in range(1,1+self.HUBS):  
            self.hubs[i-1].set_location(self.locations[i])

    def read_products(self):
        self.products=[]
        new_line=self.instance.readline()
        while new_line.strip():
            splitted_new_line=new_line.split()
            self.products.append(Product(int(splitted_new_line[0]),int(splitted_new_line[1])))
            new_line=self.instance.readline()

    def read_requests(self):
        self.requests=[]
        new_line=self.instance.readline()
        while new_line.strip():
            splitted_new_line=new_line.split()
            product_dict=dict(zip(self.products, [int(splitted_new_line[3+i]) for i in range(len(self.products))]))
            self.requests.append(Request(int(splitted_new_line[0]),int(splitted_new_line[1]),[location for location in self.locations if location.id==int(splitted_new_line[2])][0],product_dict))
            new_line=self.instance.readline()
    
    def connect_hub_to_request(self):
        for hub in self.hubs:
            for i,req in enumerate(hub.possible_request):
                hub.possible_request[i]=[request for request in self.requests if request.id==req][0]

    def read_line(self):
        for line in self.instance:
            next_line=self.read_next_line(line) 
            if next_line:
                if self.capitalized(next_line[0]):
                    self.read_fixed_vars(next_line)
                    if next_line[0]=='PRODUCTS':
                        self.read_products()
                    elif next_line[0]=='HUBS':
                        self.read_hubs()
                    elif next_line[0]=='LOCATIONS':
                        self.read_locations()
                    elif next_line[0]=='REQUESTS':
                        self.read_requests()
        self.connect_hub_to_request()
        del self.locations
        del self.products
        del self.requests
            
    def read_next_line(self,line):
        while True:
            if not line.strip(): return None
            return line.split()



new_file='Instance_1-10/Instance_1.txt'
new_instance=Instance(new_file)
new_instance.read_line()
print(new_instance.DAYS)
print(new_instance.__dict__)