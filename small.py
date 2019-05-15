import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import time
import copy

#detect close block function
def close_block_detect(list, new_element):
	if tuple([new_element[0],new_element[1]+1]) in list:
		return False
	if tuple([new_element[0],new_element[1]-1]) in list:
		return False
	if tuple([new_element[0]+1,new_element[1]+1]) in list:
		return False
	if tuple([new_element[0]+1,new_element[1]-1]) in list:
		return False
	if tuple([new_element[0]-1,new_element[1]+1]) in list:
		return False
	if tuple([new_element[0]-1,new_element[1]-1]) in list:
		return False
	if tuple([new_element[0]+1,new_element[1]]) in list:
		return False
	if tuple([new_element[0]-1,new_element[1]]) in list:
		return False
	return True
	
#recursive to find all group	
def recursive_group(map,group_map,i,j,index):
	group_flag = False
	if i != 0 and map[i-1][j] == map[i][j]:
		if group_map[i-1][j] == -1:
			group_flag = True
			group_map[i-1][j] = index
			recursive_group(map,group_map,i-1,j,index)
	if i != 4 and map[i+1][j] == map[i][j]:
		if group_map[i+1][j] == -1:
			group_flag = True
			group_map[i+1][j] = index
			recursive_group(map,group_map,i+1,j,index)
	if j != 0 and map[i][j-1] == map[i][j]:
		if group_map[i][j-1] == -1:
			group_flag = True
			group_map[i][j-1] = index
			recursive_group(map,group_map,i,j-1,index)
	if j != 4 and map[i][j+1] == map[i][j]:
		if group_map[i][j+1] == -1:
			group_flag = True
			group_map[i][j+1] = index
			recursive_group(map,group_map,i,j+1,index)
	if group_flag == True:
		group_map[i][j] = index
	return map, group_map
	
#detect available block group
def group_block(map):
	group_map = []
	index = 1
	for i in range(5):
		group_map.append([])
		for j in range(5):
			group_map[-1].append(-1)
	for i in range(5):
		for j in range(5):
			if group_map[i][j] == -1:
				if map[i][j] == 0:
					group_map[i][j] = 0
				else:
					map,group_map = recursive_group(map,group_map,i,j,index)
					if group_map[i][j] != -1:
						index = index + 1
	return group_map

#get all locate of certain label group block
def locate_block(map, label):
	result_locate = []
	for i in range(len(map)):
		part_result = [j for j,x in enumerate(map[i]) if x==label]
		for j in part_result:
			result_locate.append([i, j])
	return result_locate
	
def delete_block(map, locate):
	copy_map = copy.deepcopy(map)
	for i in locate:
		copy_map[i[0]][i[1]] = 0
	
	changed_block = []
	for i in locate:
		y, x = i[0], i[1]
		
		if y > 0 and copy_map[y-1][x] != 0 and [y-1, x] not in changed_block:
			copy_map[y-1][x] = copy_map[y-1][x] + 1
			if copy_map[y-1][x] > 4:
				copy_map[y-1][x] = 1
			changed_block.append([y-1,x])
			
		if y < 4 and copy_map[y+1][x] != 0 and [y+1, x] not in changed_block:
			copy_map[y+1][x] = copy_map[y+1][x] + 1
			if copy_map[y+1][x] > 4:
				copy_map[y+1][x] = 1
			changed_block.append([y+1,x])
			
		if x > 0 and copy_map[y][x-1] != 0 and [y, x-1] not in changed_block:
			copy_map[y][x-1] = copy_map[y][x-1] + 1
			if copy_map[y][x-1] > 4:
				copy_map[y][x-1] = 1
			changed_block.append([y,x-1])
			
		if x < 4 and copy_map[y][x+1] != 0 and [y, x+1] not in changed_block:
			copy_map[y][x+1] = copy_map[y][x+1] + 1
			if copy_map[y][x+1] > 4:
				copy_map[y][x+1] = 1
			changed_block.append([y,x+1])
	return copy_map

def flatten(seq):
	for el in seq:
		if isinstance(el, list):
			yield from flatten(el)
		else:
			yield el
	
def find_total_group(map):
	copy_map = copy.deepcopy(map)
	return(max(flatten(copy_map)))
	
#recursive to find all result
def dig_one_block(map, group_result, tap_index):
	min_single_block = 242
	min_block_label = 0
	
	total_group = find_total_group(group_result)
	for i in range(1, total_group+1):
		matches = locate_block(group_result, i) 
		
		#delete block and change label near block
		temp_map = delete_block(map, matches)

		#regroup
		temp_group_result = group_block(temp_map)
		
		#check left block
		temp_left_block = 0
		
		for j in temp_group_result:
			for k in j:
				if k == -1:
					temp_left_block = temp_left_block + 1
		if min_single_block > temp_left_block:
			min_single_block = temp_left_block
			min_block_label = i
		
	#found least single block, update to real map and group result
	matches = locate_block(group_result, min_block_label) 
		
	#delete block and change label near block
	map = delete_block(map, matches)

	#regroup
	group_result = group_block(map)
	
	#update to tap map to show tap order
	for i in matches:
		tap_map[i[0]][i[1]] = tap_order
	tap_order = tap_order + 1
	
	group_flag = False
	for i in group_result:
		for j in i:
			if j == 1:
				group_flag = True
				break
	
	return min_map, min_group, min_tap_map, min_single_block
		
	
#block detection
map_rgb = cv2.imread('55.png')
group_rgb = cv2.imread('55.png')

template_b1 = cv2.imread('block_1.png')
template_b2 = cv2.imread('block_2.png')
template_b3 = cv2.imread('block_3.png')
template_b4 = cv2.imread('block_4.png')

w_b1, h_b1 = template_b1.shape[:-1]
w_b2, h_b2 = template_b2.shape[:-1]
w_b3, h_b3 = template_b3.shape[:-1]
w_b4, h_b4 = template_b4.shape[:-1]

res_b1 = cv2.matchTemplate(map_rgb, template_b1, cv2.TM_CCOEFF_NORMED)
res_b2 = cv2.matchTemplate(map_rgb, template_b2, cv2.TM_CCOEFF_NORMED)
res_b3 = cv2.matchTemplate(map_rgb, template_b3, cv2.TM_CCOEFF_NORMED)
res_b4 = cv2.matchTemplate(map_rgb, template_b4, cv2.TM_CCOEFF_NORMED)

threshold_b1 = .92
threshold_b2 = .92
threshold_b3 = .92
threshold_b4 = .92

loc_b1 = np.where(res_b1 >= threshold_b1)
loc_b2 = np.where(res_b2 >= threshold_b2)
loc_b3 = np.where(res_b3 >= threshold_b3)
loc_b4 = np.where(res_b4 >= threshold_b4)

pure_loc_b1 = []
pure_loc_b2 = []
pure_loc_b3 = []
pure_loc_b4 = []

block_locate = []

#delete too close block
for pt in zip(*loc_b1[::-1]):
	if len(pure_loc_b1) == 0:
		pure_loc_b1.append(pt)
		block_locate.append([pt,1])
	else:
		if close_block_detect(pure_loc_b1, pt):
			pure_loc_b1.append(pt)
			block_locate.append([pt,1])

for pt in zip(*loc_b2[::-1]):
	if len(pure_loc_b2) == 0:
		pure_loc_b2.append(pt)
		block_locate.append([pt,2])
	else:
		if close_block_detect(pure_loc_b2, pt):
			pure_loc_b2.append(pt)
			block_locate.append([pt,2])

for pt in zip(*loc_b3[::-1]):
	if len(pure_loc_b3) == 0:
		pure_loc_b3.append(pt)
		block_locate.append([pt,3])
	else:
		if close_block_detect(pure_loc_b3, pt):
			pure_loc_b3.append(pt)
			block_locate.append([pt,3])

for pt in zip(*loc_b4[::-1]):
	if len(pure_loc_b4) == 0:
		pure_loc_b4.append(pt)
		block_locate.append([pt,4])
	else:
		if close_block_detect(pure_loc_b4, pt):
			pure_loc_b4.append(pt)
			block_locate.append([pt,4])
			
block_locate.sort()

for i in range(1, len(block_locate)):
	if block_locate[i][0][0] == block_locate[i-1][0][0] + 1:
		block_locate[i][0] = tuple([block_locate[i-1][0][0], block_locate[i][0][1]])

#draw block result
for i in block_locate:
	if i[1] == 1:
		cv2.rectangle(map_rgb, i[0], (i[0][0] + w_b1, i[0][1] + h_b1), (40, 40, 40), -1)
	if i[1] == 2:
		cv2.rectangle(map_rgb, i[0], (i[0][0] + w_b2, i[0][1] + h_b2), (208, 161, 251), -1)
	if i[1] == 3:
		cv2.rectangle(map_rgb, i[0], (i[0][0] + w_b3, i[0][1] + h_b3), (145, 208, 235), -1)
	if i[1] == 4:
		cv2.rectangle(map_rgb, i[0], (i[0][0] + w_b4, i[0][1] + h_b4), (46, 202, 123), -1)
print(len(block_locate))
cv2.imwrite('result.png', map_rgb)

#build label map
map = []
temp = []
for i in range(5):
	temp.append([])
	for j in range(5):
		temp[-1].append(block_locate[i*5+j])
	temp[-1].sort()

for i in range(5):
	map.append([])
	for j in range(5):
		map[-1].append(temp[j][i][1])
for i in map:
	print(i)
	
print('\n\n')
group_result = group_block(map)

#break grouped block (always break group 1 blocks)
group_flag = True
tap_map = []
for i in range(5):
	tap_map.append([])
	for j in range(5):
		tap_map[-1].append(-1)

tap_order = 1

start = time.time()
"""
while group_flag:
	#just find group with label = 1
	min_single_block = 242
	min_block_label = 0
	
	total_group = find_total_group(group_result)
	for i in range(1, total_group+1):
		matches = locate_block(group_result, i) 
		
		#delete block and change label near block
		temp_map = delete_block(map, matches)

		#regroup
		temp_group_result = group_block(temp_map)
		
		#check left block
		temp_left_block = 0
		for j in temp_group_result:
			for k in j:
				if k == -1:
					temp_left_block = temp_left_block + 1
		if min_single_block > temp_left_block:
			min_single_block = temp_left_block
			min_block_label = i
		
	#found least single block, update to real map and group result
	matches = locate_block(group_result, min_block_label) 
		
	#delete block and change label near block
	map = delete_block(map, matches)

	#regroup
	group_result = group_block(map)
	
	#update to tap map to show tap order
	for i in matches:
		tap_map[i[0]][i[1]] = tap_order
	tap_order = tap_order + 1
	
	group_flag = False
	for i in group_result:
		for j in i:
			if j == 1:
				group_flag = True
				break
"""
#result_map, result_group, result_tap, result_left = dig_one_block(map, group_result, 1)
end = time.time()

print("time elapsed:",end-start)
for i in tap_map:
	print(i)

left_block = 0
	
for i in range(5):
	for j in range(5):
		cv2.rectangle(group_rgb, temp[j][i][0], (temp[j][i][0][0] + w_b1, temp[j][i][0][1] + h_b1), (0, 0, 0), -1)
		if tap_map[i][j]>0:
			cv2.putText(group_rgb, str(tap_map[i][j]), tuple([temp[j][i][0][0], temp[j][i][0][1]+40]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
		else:
			left_block = left_block + 1
cv2.imwrite('result2.png', group_rgb)
print(left_block)