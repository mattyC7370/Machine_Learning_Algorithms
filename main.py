import numpy as np
import statistics
import pandas as pd
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def str_cmp(str_1, str_2):
    # remove whitespace
    # lowercase all characters
    # print("Here, 0", str_1, str_2)
    str_1 = ''.join(str_1.split())
    str_2 = ''.join(str_2.split())
    # print("Here, 1", str_1, str_2)
    str_1 = str_1.lower()
    str_2 = str_2.lower()
    # print("Here, 2", str_1, str_2)
    if str_1 == str_2:
        return True
    else:
        return False


def power_list(arr):
    newArray = []
    newArray = [0 for i in range(len(arr))]
    newArraySummation = 0
    for i in range(len(arr)):
        newArray[i] = arr[i] ** i

    for i in range(len(arr)):
        newArraySummation = newArraySummation + newArray[i]

    return newArraySummation


def create_stats(universities):
    """
    Return a tuple of (mean students enrolled, median students enrolled, mean tuition, median tuition)
    Return a tuple of (mean cars sold, median cars sold, mean MSRP, median MSRP)
    """
    totalStudentsEnrolled = 0
    totalTuition = 0
    studentsEnrolledArray = [0 for i in range(len(universities))]
    tuitionArray = [0 for i in range(len(universities))]
    for i in range(len(universities)):
        totalStudentsEnrolled = totalStudentsEnrolled + (universities[i])[1]
        totalTuition = totalTuition + (universities[i])[2]
        studentsEnrolledArray[i] = (universities[i])[1]
        tuitionArray[i] = (universities[i])[2]

    returnTuple = (totalStudentsEnrolled / len(universities), statistics.median(studentsEnrolledArray),
                   totalTuition / len(universities), statistics.median(tuitionArray))
    return returnTuple


def dot_product(mat_1, mat_2):
    return np.dot(mat_1, mat_2)


def cross_product(mat_1, mat_2):
    np.cross(mat_1, mat_2)


def inverse(mat):
    return np.linalg.inv(mat)


def subtract_mean(mat):
    # for each row, get mean, subtract drom each element
    newMat = np.array(mat, dtype=float)

    for i in range(len(mat)):
        localMean = np.mean(mat[i])
        newMat[i] = mat[i] - localMean
        # for j in range(len(mat[i])):
        #     (newMat[i])[j] = (float((mat[i])[j]) - float(localMean))
        #     # subtracting 2 instead of 2.5????

    return newMat


def matrix_reshape(mat, n_rows, n_cols):
    a = np.reshape(mat, (n_rows, n_cols))
    return a


def moving_average(arr, window_size):
    newArr = np.arange(len(arr) - (window_size - 1), dtype=float)
    for i in range(len(newArr)):
        newArr[i] = arr[i:i + window_size].mean()

    return newArr


if __name__ == '__main__':
    print("hello world")
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()

