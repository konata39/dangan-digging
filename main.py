import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

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

#block detection
map_rgb = cv2.imread('7.png')

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


