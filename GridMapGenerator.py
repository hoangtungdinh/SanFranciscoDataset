import re
import matplotlib.pyplot as plt
import numpy as np

import math

DEPTH = 14
WIDTH = 2 ** DEPTH
SCATTER_PLOT = False
GRID_PLOT = False
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

print(x_res, y_res)

f = open('sanfrancisco.txt', 'w')
f.write('%d\n' % DEPTH)
f.write('%d\t%d\n' % (round(x_res), round(y_res)))

for x in range(WIDTH):
    for y in range(WIDTH):
        f.write('%d %d %d\n' % (x, y, map[x][y]))
f.close()

if SCATTER_PLOT:
    plt.figure(1)
    plt.scatter(x_base, y_base, color='blue')
    plt.scatter(x_filtered, y_filtered, color='red')

if GRID_PLOT:
    numpy_map = np.array(map)
    plt.figure(2)
    plt.imshow(np.rot90(numpy_map), interpolation='nearest')
    plt.grid(True)

if SCATTER_PLOT or GRID_PLOT:
    plt.show()