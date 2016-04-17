import random
import re
import matplotlib.pyplot as plt
import numpy as np

import math


def valid_pair(map, resolution, origin, destination):
    if map[origin[0]][origin[1]] == 1:
        return False

    if map[destination[0]][destination[1]] == 1:
        return False

    point_origin = (origin[0]*resolution[0], origin[1]*resolution[1])
    point_destination = (destination[0]*resolution[0], destination[1]*resolution[1])
    distance = math.sqrt((point_origin[0] - point_destination[0])**2 + (point_origin[1] - point_destination[1])**2)
    width = len(map)
    min_distance = math.sqrt((width*resolution[0])**2 + (width*resolution[1])**2) / 2 # half of the diagonal path

    if distance < min_distance:
        return False

    return True

if __name__ == '__main__':
    DEPTH = 11
    WIDTH = 2 ** DEPTH
    SCATTER_PLOT = True
    GRID_PLOT = False
    OD_PLOT = False
    WRITE_TO_FILE = False
    GENERATE_OD = 30
    ALTITUDE = 30000 # millimeters

    fileName = 'san_francisco.csv'

    data = []

    lambda_0 = float('inf')

    with open(fileName) as f:
        next(f)
        for line in f:
            list_string = re.split(",|\n", line.replace('\n', ''))
            list_double = [float(x) for x in list_string]
            data.append(list_double)
            if lambda_0 > min(list_double[1], list_double[3]):
                lambda_0 = min(list_double[1], list_double[3])

    lambda_radian_0 = math.radians(lambda_0)

    R = 6371000 * 1000  # Earth radius in millimeters

    dat_xy = []
    y_min = float('inf')

    for point in data:
        # everything now is in millimeters
        x0 = R * (math.radians(point[1]) - lambda_radian_0)
        y0 = R * math.log(math.tan(math.pi / 4 + math.radians(point[0]) / 2))

        x1 = R * (math.radians(point[3]) - lambda_radian_0)
        y1 = R * math.log(math.tan(math.pi / 4 + math.radians(point[2]) / 2))

        height = point[4] * 0.3048 * 1000 # feet to millimeters

        dat_xy.append((x0, y0, x1, y1, height))

        if y_min > min(y0, y1):
            y_min = min(y0, y1)

    x_base = []
    y_base = []
    x_filtered = []
    y_filtered = []

    dat_normalized = []

    x_diff = 0
    y_diff = 0

    for point in dat_xy:
        x0 = point[0]
        y0 = point[1] - y_min

        x1 = point[2]
        y1 = point[3] - y_min

        height = point[4]

        dat_normalized.append((x0, y0, x1, y1, height))

        if x_diff < max(x0, x1):
            x_diff = max(x0, x1)

        if y_diff < max(y0, y1):
            y_diff = max(y0, y1)

        x_base.append(x0)
        x_base.append(x1)
        y_base.append(y0)
        y_base.append(y1)

        if height > ALTITUDE:
            x_filtered.append(x0)
            x_filtered.append(x1)
            y_filtered.append(y0)
            y_filtered.append(y1)

    x_res = x_diff / WIDTH
    y_res = y_diff / WIDTH

    map = [[0 for i in range(WIDTH)] for i in range(WIDTH)]

    for square in dat_normalized:
        if square[4] <= ALTITUDE:
            continue

        x_min = min(square[0], square[2]) / x_res
        x_max = max(square[0], square[2]) / x_res
        y_min = min(square[1], square[3]) / y_res
        y_max = max(square[1], square[3]) / y_res

        for x in range(int(x_min), int(x_max) + 1):
            for y in range(int(y_min), int(y_max) + 1):
                map[x][y] = 1

    if WRITE_TO_FILE:
        f = open('sanfrancisco.txt', 'w')
        content = ""
        content += '%d\n' % DEPTH
        content += '%d %d\n' % (round(x_res), round(y_res))

        for x in range(WIDTH):
            for y in range(WIDTH):
                content += '%d %d %d\n' % (x, y, map[x][y])

        f.write(content)
        f.close()

    if GENERATE_OD > 0:
        od = []
        random.seed(111)
        for i in range(GENERATE_OD):
            row_origin = random.randrange(WIDTH)
            col_origin = random.randrange(WIDTH)
            row_destination = random.randrange(WIDTH)
            col_destination = random.randrange(WIDTH)

            while not valid_pair(map, (round(x_res), round(y_res)), (row_origin, col_origin),
                                 (row_destination, col_destination)):
                row_origin = random.randrange(WIDTH)
                col_origin = random.randrange(WIDTH)
                row_destination = random.randrange(WIDTH)
                col_destination = random.randrange(WIDTH)

            od.append((row_origin*round(x_res), col_origin*round(y_res), row_destination*round(x_res),
                       col_destination*round(y_res)))

            if OD_PLOT:
                map[row_origin][col_origin] = 0.5
                map[row_destination][col_destination] = 0.5

            pairFile = open('SanFranciscoODs.txt', 'w')
            pair_content = ""
            for pair in od:
                pair_content += '%d %d %d %d\n' % (pair[0], pair[1], pair[2], pair[3])
            pairFile.write(pair_content)
            pairFile.close()

    if SCATTER_PLOT:
        plt.figure(1)
        plt.scatter(x_base, y_base, color='blue')
        plt.scatter(x_filtered, y_filtered, color='red')

    if GRID_PLOT:
        numpy_map = np.array(map)
        plt.figure(2)
        plt.imshow(np.rot90(numpy_map), interpolation='nearest')
        plt.grid(True)
        plt.savefig('gridmap.pdf', format='pdf', dpi=1000)