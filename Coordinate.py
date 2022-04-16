import math
class Coordinate:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._to_hub={}
        self._cost_dist_to_hub=[]

    def get_cor(self):
        return (self._x , self._y)

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_to_hub(self):
        return self._to_hub

    def set_dist_to_hub(self,hubs,VAN_COST):
        for key,value in hubs.items():
            dist=math.floor(math.sqrt((self._x-value['location'].get_x())**2+(self._y-value['location'].get_y())**2))
            self._to_hub[key]={'distance':dist}
            self._to_hub[key]['cost']=VAN_COST*dist


        