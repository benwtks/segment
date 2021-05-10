import sys
import math
from random import sample, randint
import numpy as np
from skimage import io
import itertools

def get_segments(segmented, classes):
    shape = segmented.shape
    segments = np.zeros((classes, shape[0], shape[1]))
    last_index = 0
    colours_seen = [segmented[0][0]]

    for y in range(shape[1]):
        for x in range(shape[0]):
            if segmented[y][x] != colours_seen[last_index]:
                try:
                    last_index = colours_seen.index(segmented[y][x])
                except ValueError:
                    colours_seen.append(segmented[y][x])
                    last_index = len(colours_seen) - 1
            segments[last_index][y][x] = 255

    return segments

def jaccard(shape, segments1, segments2):
    if len(segments1) != len(segments2):
        raise ValueError("Segments of different length")

    # Which segments in 1 did each of the segments in 2 match to
    max_jaccards = [[] for i in range(len(segments1))]
    # What are the jaccards of each segment in 1 compared to each in 2
    jaccards = [[] for j in range(len(segments2))]

    for i in range(len(segments1)):
        # (j of segment, jaccard compared to j)
        max_jaccard = (None, 0)
        for j in range(len(segments2)):
            intersection = union = 0
            for (x1, y1) in itertools.product(range(shape[1]), range(shape[0])):
                if segments1[i][y1][x1] == 255:
                    if segments2[j][y1][x1] == 255:
                        intersection += 1
                    union += 1
                    continue

                if segments2[j][y1][x1] == 255:
                    union += 1

            jaccard = intersection / union
            jaccards[i].append((j, jaccard))
            if jaccard > max_jaccard[1]:
                max_jaccard = (j, jaccard)

        max_jaccards[max_jaccard[0]].append(i)

    for i in range(len(segments1)):
        jaccards[i].sort(key=(lambda val: val[1]), reverse=True)

    while True:
        complete = True
        for j in range(len(segments2)):
            while len(max_jaccards[j]) > 1:
                lowest = (max_jaccards[j][0], jaccards[max_jaccards[j][0]][0][1])
                for i in max_jaccards[j]:
                    if jaccards[i][0][1] < lowest[1]:
                        lowest = (i, jaccards[i][0][1])

                # deallocate lowest and reallocate it to its next highest
                max_jaccards[j].remove(lowest[0])
                del jaccards[lowest[0]][0]
                new_alloc = jaccards[lowest[0]][0][0]
                if len(max_jaccards[new_alloc]) > 1:
                    complete = False

                max_jaccards[new_alloc].append(lowest[0])

        if complete:
            break

    total_jaccard = 0
    for i in range(len(jaccards)):
        total_jaccard += jaccards[i][0][1]

    accuracy = round(total_jaccard / len(segments1), 2)
    return accuracy


classes = int(sys.argv[1])
image1 = io.imread(sys.argv[2], as_gray=True)
image2 = io.imread(sys.argv[3], as_gray=True)

segments1 = get_segments(image1, classes)
segments2 = get_segments(image2, classes)

print(jaccard(image1.shape, segments1, segments2))
