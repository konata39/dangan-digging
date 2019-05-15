import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import time

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
	if i != 10 and map[i+1][j] == map[i][j]:
		if group_map[i+1][j] == -1:
			group_flag = True
			group_map[i+1][j] = index
			recursive_group(map,group_map,i+1,j,index)
	if j != 0 and map[i][j-1] == map[i][j]:
		if group_map[i][j-1] == -1:
			group_flag = True
			group_map[i][j-1] = index
			recursive_group(map,group_map,i,j-1,index)
	if j != 21 and map[i][j+1] == map[i][j]:
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
	for i in range(11):
		group_map.append([])
		for j in range(22):
			group_map[-1].append(-1)
	for i in range(11):
		for j in range(22):
			if group_map[i][j] == -1:
				map,group_map = recursive_group(map,group_map,i,j,index)
				if group_map[i][j] != -1:
					index = index + 1
	return group_map

#block detection
map_rgb = cv2.imread('7.png')
group_rgb = cv2.imread('7.png')

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
		cv2.rectangle(map_rgb, i[0], (i[0][0] + w_b1, i[0][1] + h_b1), (100, 100, 100), -1)
	if i[1] == 2:
		cv2.rectangle(map_rgb, i[0], (i[0][0] + w_b2, i[0][1] + h_b2), (208, 161, 251), -1)
	if i[1] == 3:
		cv2.rectangle(map_rgb, i[0], (i[0][0] + w_b3, i[0][1] + h_b3), (145, 208, 235), -1)
	if i[1] == 4:
		cv2.rectangle(map_rgb, i[0], (i[0][0] + w_b4, i[0][1] + h_b4), (216, 202, 123), -1)
print(len(block_locate))
cv2.imwrite('result.png', map_rgb)

#build label map
map = []
temp = []
for i in range(22):
	temp.append([])
	for j in range(11):
		temp[-1].append(block_locate[i*11+j])
	temp[-1].sort()

for i in range(11):
	map.append([])
	for j in range(22):
		map[-1].append(temp[j][i][1])
for i in map:
	print(i)
	
print('\n\n')
group_result = group_block(map)

for i in range(11):
	for j in range(22):
		cv2.rectangle(group_rgb, temp[j][i][0], (temp[j][i][0][0] + w_b1, temp[j][i][0][1] + h_b1), (0, 0, 0), -1)
		if group_result[i][j]>0:
			cv2.putText(group_rgb, str(group_result[i][j]), tuple([temp[j][i][0][0], temp[j][i][0][1]+40]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite('result2.png', group_rgb)
end = time.time()
