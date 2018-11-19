import math
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
from threading import Thread
import matlab.engine
import os


def is_valid_pt(point, max_height, max_width):
    return (0 <= point[0] < max_height) and (0 <= point[1] < max_width)


def find_neighbors(point):
    row = point[0]
    col = point[1]
    top = (row - 1, col)
    bot = (row + 1, col)
    left = (row, col - 1)
    right = (row, col + 1)
    top_left = (row - 1, col - 1)
    top_right = (row - 1, col + 1)
    bot_left = (row + 1, col - 1)
    bot_right = (row + 1, col + 1)
    return top, bot, left, right, top_left, top_right, bot_left, bot_right


def find_nearest_point(ori_point, pixel_set, max_height, max_width):
    visited = set()
    # The map for finding the next best point
    dist_map = {0.0: {ori_point}}
    while dist_map:
        shortest_dist = min(dist_map)
        next_pt = dist_map[shortest_dist].pop()
        visited.add(next_pt)
        if len(dist_map[shortest_dist]) == 0:
            del dist_map[shortest_dist]
        if next_pt in pixel_set:
            return next_pt
        neighbors = find_neighbors(next_pt)
        for neighbor in neighbors:
            if is_valid_pt(neighbor, max_height, max_width) and neighbor not in visited:
                dist = math.sqrt((neighbor[0] - ori_point[0]) ** 2 + (neighbor[1] - ori_point[1]) ** 2)
                if dist < 15:
                    if dist not in dist_map:
                        dist_map[dist] = {neighbor}
                    else:
                        dist_map[dist].add(neighbor)
    return None


def find_nearest(edges, max_height, max_width):
    new_edges = set()
    pixel_set = set()
    # Loads in all pixels
    for edge in edges:
        for pixel in edge:
            pixel_set.add(tuple(pixel))
    for edge in edges:
        temp_list = []
        # Remove points that on the edge
        for pixel in edge:
            pixel = tuple(pixel)
            if pixel in pixel_set:
                pixel_set.remove(pixel)
            temp_list.append(pixel)
        end_pt1 = tuple(edge[0])
        end_pt2 = tuple(edge[len(edge) - 1])
        # TODO: expect point to be (352, 429)
        nearest1 = find_nearest_point(end_pt1, pixel_set, max_height, max_width)
        nearest2 = find_nearest_point(end_pt2, pixel_set, max_height, max_width)
        if nearest1 is not None:
            edge1 = (end_pt1, nearest1)
            new_edges.add(edge1)
        if nearest2 is not None:
            edge2 = (end_pt2, nearest2)
            new_edges.add(edge2)
        # Recover points back to all possible pixels
        for pixel in temp_list:
            pixel_set.add(tuple(pixel))
    return new_edges


def get_end_points(edge):
    end_point1 = edge[0]
    end_point2 = edge[len(edge) - 1]
    return tuple(end_point1), tuple(end_point2)


def add_element_to_map(map_structure, key, value):
    if key in map_structure:
        map_structure[key].append(value)
    else:
        map_structure[key] = [value]


def map_skeleton_with_branching_points(branching_points, skeletons):
    branching_points_map = {}
    end_points_map = {}
    for skeleton in skeletons:
        end_point1, end_point2 = get_end_points(skeleton)
        add_element_to_map(end_points_map, end_point1, skeleton)
        add_element_to_map(end_points_map, end_point2, skeleton)
    for branching_point in branching_points:
        branching_point = tuple(branching_point)
        if branching_point in end_points_map:
            temp_skeletons = end_points_map[branching_point]
            for temp_skeleton in temp_skeletons:
                add_element_to_map(branching_points_map, branching_point, temp_skeleton)
        else:
            neighbors = find_neighbors(branching_point)
            for neighbor in neighbors:
                if neighbor in end_points_map:
                    temp_skeletons = end_points_map[neighbor]
                    for temp_skeleton in temp_skeletons:
                        add_element_to_map(branching_points_map, branching_point, temp_skeleton)
    return branching_points_map


def filter_branches(branching_point_map):
    is_severely_branched = False
    merged_branch_set = set()
    for key in branching_point_map:
        branches = branching_point_map[key]
        if len(branches) <= 3:
            min_len = len(branches[0])
            min_index = 0
            for n, branch in enumerate(branches):
                if len(branch) < min_len:
                    min_len = len(branch)
                    min_index = n
            del branches[min_index]
        else:
            is_severely_branched = True
            break
    if not is_severely_branched:
        for key in branching_point_map:
            branches = branching_point_map[key]
            for branch in branches:
                for point in branch:
                    point = tuple(point)
                    merged_branch_set.add(point)
    return is_severely_branched, merged_branch_set


def extracting_brush_strokes(_eng_, _ori_file_names_, _result_file_names_):
    for painting_name, result_name in zip(_ori_file_names_, _result_file_names_):
        _eng_.edgedetection(painting_name, nargout=0)
        mat = scipy.io.loadmat('./Alpha/step1_edgelist.mat')
        edge_list = mat['edgelist'][0][2][0]
        height = mat['edgelist'][0][0][0]
        width = mat['edgelist'][0][1][0]
        img = np.zeros((int(height), int(width)), np.uint8)
        for _edge in edge_list:
            for _pixel in _edge:
                img[_pixel[0] - 1][_pixel[1] - 1] = 1
        plt.imshow(img, cmap='gray')
        plt.show()
        # TODO: (354, 417)
        # ((354, 417), (357, 415))
        new_edges_gen = find_nearest(edge_list, height, width)
        for _edge in new_edges_gen:
            for index in range(0, len(_edge) - 1):
                point1 = _edge[index]
                point1 = (point1[1], point1[0])
                point2 = _edge[index + 1]
                point2 = (point2[1], point2[0])
                img = cv2.line(img, point1, point2, 1, 1)
        scipy.io.savemat('../edge/Alpha/step1_img', {'im2': img})
        _eng_.step2(nargout=0)

        branching_point_mat = scipy.io.loadmat('./Alpha/step2_junction.mat')
        _branching_points = branching_point_mat['junction'][0]
        edge_mat = scipy.io.loadmat('./Alpha/step2_theedgelist.mat')
        edges = edge_mat['theedgelist'][0]

        # The result array will contain all valid brush stroke skeletons
        result = []
        severely_branch = []
        for _n, _edge in enumerate(edges):
            if len(_edge) > 0:
                _skeletons = _edge[0]
                if len(_branching_points[_n]) > 0:
                    _branching_point_map = map_skeleton_with_branching_points(_branching_points[_n], _skeletons)
                    _is_severely_branched, _merged_branch_set = filter_branches(_branching_point_map)
                    severely_branch.append(_is_severely_branched)
                else:
                    _merged_branch_set = set()
                    for _skeleton in _skeletons:
                        for _point in _skeleton:
                            _point = tuple(_point)
                            _merged_branch_set.add(_point)
                    _is_severely_branched = False
                    severely_branch.append(_is_severely_branched)
                if not _is_severely_branched:
                    result.append(_merged_branch_set)
            else:
                severely_branch.append(True)
        for i in range(0, len(result)):
            result[i] = list(result[i])

        scipy.io.savemat('../edge/Alpha/step3_severely_branch', {'severely_branch': severely_branch})
        scipy.io.savemat('../edge/Alpha/step3_result.mat', {'result': result})

        _eng_.after_first_judge(result_name, nargout=0)


def main():
    eng = matlab.engine.start_matlab()
    eng.addpath('../edge/Alpha')
    eng = matlab.engine.start_matlab()

    base = 'input/wikipainting/'  # TODO: this line should be adjusted to match with the path
    output_folder = 'output/'

    folders = os.listdir(base)

    ori_file_names = []
    result_file_names = []
    for folder in folders:
        if folder != '.DS_Store':
            if not os.path.isdir(output_folder + folder):
                os.makedirs(output_folder + folder)
            for file in os.listdir(base + folder + '/'):
                ori_file_names.append(base + folder + '/' + file)
                result_file_names.append(output_folder + folder + '/' + file)

    cores = 4
    num_files_per_core = round(len(ori_file_names) / cores)

    ori_groups = []
    result_groups = []
    # Split the data for different threads
    for i in range(0, cores):
        start_index = i * num_files_per_core
        if i < cores - 1:
            ori_group = ori_file_names[start_index:start_index + num_files_per_core]
            result_group = result_file_names[start_index:start_index + num_files_per_core]
        else:
            ori_group = ori_file_names[start_index:]
            result_group = result_file_names[start_index:]
        ori_groups.append(ori_group)
        result_groups.append(result_group)

    threads = []
    for i in range(0, cores):
        ori_group = ori_groups[i]
        result_group = result_groups[i]
        thread = Thread(target=extracting_brush_strokes, args=(eng, ori_group, result_group))
        print('starting the extracting brush stroke thread ' + str(i))
        thread.start()
        threads.append(thread)
    for n, thread in enumerate(threads):
        thread.join()
        print('finished extracting brush stroke thread ' + str(n))


if __name__ == '__main__':
    print('starting the brush stroke extraction...')
    main()
    print('finished brush stroke extraction')
