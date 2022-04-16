class Hub:
    def __init__(self,hid,cost,pos_req):
        self.id=hid
        self.cost=cost
        self.loc=None
        self.closest_customers = []
        self.possible_request = pos_req

    def __repr__(self):
        return str(self.id)

    def __eq__(self, other):
        return self.id == other.id
    
    def set_location(self,location):
        self.loc = location


class Depot:
    def __init__(self,did,location):
        self.id=did
        self.loc=location

    def __repr__(self):
        return str(self.id)


class Customer:
    def __init__(self, cid, coordinate):
        self.id = cid
        self.coordinate = coordinate
    
    def __repr__(self):
        return str(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

class Coordinate:
    def __init__(self, id, x, y):
        self.id=id
        self.x = x
        self.y = y

    def __repr__(self):
        return str(self.id)

    def get_cor(self):
        return (self.x , self.y)
    
    def __eq__(self, other):
        return (self.get_cor() == other.get_cor())

class Request:
    def __init__(self, rid, day, costumer, demand):
        self.id = rid
        self.day = day
        self.costumer = costumer
        self.demand = demand
    
    def __repr__(self):
        return str(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

class Product():
    def __init__(self, pid, days_fresh):
        self.id = pid
        self.days_fresh = days_fresh

    def __repr__(self):
        return str(self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id