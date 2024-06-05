class Centroid:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_old = 0
        self.y_old = 0
        self.points = []
        self.cluster_number = 'no_num'

    def clear(self):
        self.coordinates = []

    def set_values(self, x_val, y_val):
        self.x_old = self.x
        self.y_old = self.y
        self.x = x_val
        self.y = y_val

    def add_point(self, x_val, y_val):
        self.coordinates.append([x_val, y_val])

    def clear_coordinates(self):
        self.coordinates = []       