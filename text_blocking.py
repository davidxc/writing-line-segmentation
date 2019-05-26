"""
This code is an adapted + extended
version of the blocking algorithm described in
'Text Binarization in Color Documents' by
Efthimios Badekas, Nikos Nikolaou, and
Nikos Papamarkos. 

There are 3 levels of blocks that we want:

1) Character blocks (corresponding to
connected components or contours)

2) Text blocks (corresponding to words, which will
be connected connected components, as described
in Text Blocking 1.pdf)

3) Line blocks (which involves grouping text blocks
into line blocks while accounting for slant. Possibly
use a greedy algorithm for this that starts with
a start point for each potential line and then follows
the text blocks? I think this would work for slanted
lines. Stop when there are no more text blocks
that are to the right and close enough.)
"""

"""
Important differences from the algorithm outlined in
the paper. At this point, I've made enough changes
that it makes sense to describe it as an adapted version of the
algorithm in Text Blocking 1.pdf.

I'm applying dilation with a (2, 2) kernel before finding
contours.

I'm applying a threshold of at least 20 pixels for a
contour before including that contour

Using a value of T_PSR=11 instead of 7 as in the paper
to account for background writing being connected to a character
and thus increasing the ratio. A good test might be to also
do the three levels of blocking AFTER removing the faint
writing and compare it to the blocking achieved before
removing the faint writing.

We require vertical overlap between the character blocks when creating
the text blocks (text blocks = "level 2" blocks)

We compute "line blocks" as done in the
function create_level_3_boxes. The paper
says something about their blocks (which are really
the level 2 blocks here) corresponding to lines,
but I don't see how that can be the case unless
they have documents with a small width (such that
a line is only a word or two). Their groups
of connected contours in the paper are my level
2 blocks.

We also merge small (by area) line blocks into the closest large one. This merging
step makes sense as a step to do before binarizing by line blocks.
"""

import cv2
import numpy as np
import math
import sys
import copy

# the c_d in d_max = c_d * max(W_i, H_i)
# in Text Blocking 1.pdf
CONSTANT_C_D = float(3)
CONSTANT_C_D_GROUPS = CONSTANT_C_D
# d_min as in d_min <= d_i <= d_max
# in Text Blocking 1.pdf
CONSTANT_D_MIN = 5

# T_PSR is the value used in Equation (6) of Text Blocking 1.pdf
T_PSR = float(11)

"""
we merge a line group if it's size (total area of its contours)
is less than THRESH_LINE_GROUP_SIZE_MULT * (size of the largest line group)
"""
THRESH_LINE_GROUP_SIZE_MULT = float(0.20)

def get_blocks(binary_output_img, color_img, original_file_name):
	"""
	
	Takes as input a binary img
	and returns character blocks, text blocks,
	and line blocks

	I'm applying a dilation operation to binary_output_img
	before passing it to get_contour_bounding_boxes_list. While this
	may do things like undesirably fill in holes in letters (in d for example),
	that's fine. I get more "complete" contours and contour bounding
	boxes that are around the actual characters, instead of subparts
	of characters. This is important. What I can then do is just
	take the contour points and the associated bounding boxes
	and then use the original binary_output_img to just get the
	black pixels when comparing contours.
	Actually, just applying dilation in get_contour_bounding_boxes_list
	"""
	
	# below is used as a global cache to avoid calling euclid_dist
	# unnecessarily. the key is a pair of contour ids, where ids are the keys in
	# contour_id_to_data_dict, and the value is the euclidean distance between the centroids
	# of the bounding boxes of the corresponding contours. a pair of contour ids
	# is represented as a frozenset

	CENTROID_DISTANCES_CACHE_DICT = {}

	binary_output_img_orig = np.copy(binary_output_img)
	color_img_orig = np.copy(color_img) 

	# gets "Level 1" boxes corresponding to connected components aka contours
	# along with the contour information 
	contour_bounding_boxes_list = get_contour_bounding_boxes_list(binary_output_img, color_img)

	# create fresh copies
	binary_output_img = np.copy(binary_output_img_orig)
	color_img_orig = np.copy(color_img_orig)

	"""
	get level 2

	Text blocks (corresponding to words, which will
	be connected connected components, as described
	in Text Blocking 1.pdf)
	"""
	
	# The below function modifies contour_bounding_boxes_list in place
	# For each sublist representing a contour's info in contour_bounding_boxes_list,
	# it just appends the list of connected contours to that sublist
	# note that the list of connected contours is just a list of lists, where each
	# sublist represents a contour's information (so each sublist is just a top level
	# list from contour_bounding_boxes_list)
	get_text_blocks_list(binary_output_img, color_img_orig, contour_bounding_boxes_list,
		CENTROID_DISTANCES_CACHE_DICT)

	# converting the above contour_bounding_boxes_list into a dictionary form, 
	# which I think I should have done in the first place to keep things cleaner
	# keys are the contour IDs

	contour_id_to_data_dict = {}
	for c in contour_bounding_boxes_list:
		contour_id_to_data_dict[c[4]] = {}
		# list of points that make up the contour
		"""
		note that c[0] is the c in
		for c in contours...

		where contours = cv2.findContours(...)
		so c[0] should be the cnt that we need
		to pass to cv2.pointPolygonTest when
		checking whether a point is in a contour
		cv2.pointPolygonTest(cnt, (50, 50), False)
		see 
		https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
		"""
		contour_id_to_data_dict[c[4]]["contour_points"] = c[0]
		# bounding box info, (x, y, w, h) where x and y are the top left corner
		contour_id_to_data_dict[c[4]]["contour_bounding_b_info"] = c[1]		
		# centroid of the bounding box as a tuple (x, y)
		contour_id_to_data_dict[c[4]]["centroid"] = c[2]		
		# area of the contour, NOT of the bounding box
		contour_id_to_data_dict[c[4]]["area"] = c[3]
		# list of contour IDs that this contour is connected to. 
		contour_id_to_data_dict[c[4]]["connected_contours"] = []
		for c_prime in c[5]:
			contour_id_to_data_dict[c[4]]["connected_contours"].append(c_prime[4])

	# have now finished creating contour_id_to_data_dict

	# create fresh copies for debugging
	binary_output_img = np.copy(binary_output_img_orig)
	color_img_orig = np.copy(color_img_orig)

	# debug_connections_contour_id_to_data_dict(contour_id_to_data_dict, binary_output_img)

	# level 3 boxes are ideally "line blocks"
	# line_groups_dict here is a dict where the keys are line group IDs
	# and the value is a set of contour ids that make up that line group
	# where a contour id is just an id from the contour_id_to_data_dict
	line_groups_dict = create_level_3_boxes(contour_id_to_data_dict, CENTROID_DISTANCES_CACHE_DICT)

	# below is a check that can later be taken out.
	# yeah, now there's no overlap
	# check_for_overlap(line_groups_dict)
	# debug_line_groups_dict(line_groups_dict, contour_id_to_data_dict, color_img_orig)
	# debug_line_groups_dict_one_group_at_time(line_groups_dict, contour_id_to_data_dict, color_img_orig)

	line_groups_dict = merge_small_line_groups(line_groups_dict, contour_id_to_data_dict)
	# below line should be unnecessary
	line_groups_dict = filter_line_groups_dict(line_groups_dict)

	# create fresh copies for debugging
	binary_output_img = np.copy(binary_output_img_orig)
	color_img_orig = np.copy(color_img_orig)

	# debug_line_groups_dict(line_groups_dict, contour_id_to_data_dict, color_img_orig)
	# debug_line_groups_dict_one_group_at_time(line_groups_dict, contour_id_to_data_dict, color_img_orig, original_file_name)

	return contour_id_to_data_dict, line_groups_dict

def get_blocks_use_level_2_for_3(binary_output_img, color_img, original_file_name):
	"""
	Takes as input binary output img
	(or another binary output img)
	and returns character blocks, text blocks,
	and line blocks

	I'm applying a dilation operation to binary_output_img
	before passing it to get_contour_bounding_boxes_list. While this
	may do things like undesirably fill in holes in letters (in d for example),
	that's fine. I get more "complete" contours and contour bounding
	boxes that are around the actual characters, instead of subparts
	of characters. This is important. What I can then do is just
	take the contour points and the associated bounding boxes
	and then use the original binary_output_img to just get the
	black pixels when comparing contours.
	Actually, just applying dilation in get_contour_bounding_boxes_list
	"""
	
	# below is used as a global cache to avoid calling euclid_dist
	# unnecessarily. the key is a pair of contour ids, where ids are the keys in
	# contour_id_to_data_dict, and the value is the euclidean distance between the centroids
	# of the bounding boxes of the corresponding contours. a pair of contour ids
	# is represented as a frozenset

	CENTROID_DISTANCES_CACHE_DICT = {}

	binary_output_img_orig = np.copy(binary_output_img)
	color_img_orig = np.copy(color_img) 

	# gets "Level 1" boxes corresponding to connected components aka contours
	# along with the contour information 
	contour_bounding_boxes_list = get_contour_bounding_boxes_list(binary_output_img, color_img)

	# create fresh copies
	binary_output_img = np.copy(binary_output_img_orig)
	color_img_orig = np.copy(color_img_orig)

	"""
	get level 2

	Text blocks (corresponding to words, which will
	be connected connected components, as described
	in Text Blocking 1.pdf)
	"""
	
	# The below function modifies contour_bounding_boxes_list in place
	# For each sublist representing a contour's info in contour_bounding_boxes_list,
	# it just appends the list of connected contours to that sublist
	# note that the list of connected contours is just a list of lists, where each
	# sublist represents a contour's information (so each sublist is just a top level
	# list from contour_bounding_boxes_list)
	get_text_blocks_list(binary_output_img, color_img_orig, contour_bounding_boxes_list,
		CENTROID_DISTANCES_CACHE_DICT)

	# converting the above contour_bounding_boxes_list into a dictionary form, 
	# which I think I should have done in the first place to keep things cleaner
	# keys are the contour IDs

	contour_id_to_data_dict = {}
	for c in contour_bounding_boxes_list:
		contour_id_to_data_dict[c[4]] = {}
		# list of points that make up the contour
		contour_id_to_data_dict[c[4]]["contour_points"] = c[0]
		# bounding box info, (x, y, w, h) where x and y are the top left corner
		contour_id_to_data_dict[c[4]]["contour_bounding_b_info"] = c[1]		
		# centroid of the bounding box as a tuple (x, y)
		contour_id_to_data_dict[c[4]]["centroid"] = c[2]		
		# area of the contour, NOT of the bounding box
		contour_id_to_data_dict[c[4]]["area"] = c[3]
		# list of contour IDs that this contour is connected to. 
		contour_id_to_data_dict[c[4]]["connected_contours"] = []
		for c_prime in c[5]:
			contour_id_to_data_dict[c[4]]["connected_contours"].append(c_prime[4])

	# have now finished creating contour_id_to_data_dict
	# print("Finished creating contour_id_to_data_dict")

	# create fresh copies for debugging
	binary_output_img = np.copy(binary_output_img_orig)
	color_img_orig = np.copy(color_img_orig)

	# debug_connections_contour_id_to_data_dict(contour_id_to_data_dict, binary_output_img)

	# level 3 boxes are ideally "line blocks"
	# line_groups_dict here is a dict where the keys are line group IDs
	# and the value is a set of contour ids that make up that line group
	# where a contour id is just an id from the contour_id_to_data_dict
	line_groups_dict = {}
	for k, v in contour_id_to_data_dict.iteritems():
		connected_contours_list = copy.deepcopy(v["connected_contours"])
		connected_contours_list.append(k)
		line_groups_dict[k] = set(connected_contours_list)

	# below is a check that can later be taken out.
	# yeah, now there's no overlap
	# check_for_overlap(line_groups_dict)
	# debug_line_groups_dict(line_groups_dict, contour_id_to_data_dict, color_img_orig)
	# debug_line_groups_dict_one_group_at_time(line_groups_dict, contour_id_to_data_dict, color_img_orig)

	# line_groups_dict = merge_small_line_groups(line_groups_dict, contour_id_to_data_dict)	
	line_groups_dict = filter_line_groups_dict(line_groups_dict)	
	# below line should be unnecessary
	# line_groups_dict = filter_line_groups_dict(line_groups_dict)

	# create fresh copies for debugging
	binary_output_img = np.copy(binary_output_img_orig)
	color_img_orig = np.copy(color_img_orig)

	# debug_line_groups_dict(line_groups_dict, contour_id_to_data_dict, color_img_orig)
	debug_line_groups_dict_one_group_at_time(line_groups_dict, contour_id_to_data_dict, color_img_orig, original_file_name)

	return contour_id_to_data_dict, line_groups_dict

def check_for_overlap(line_groups_dict):
	for k, v in line_groups_dict.iteritems():
		for x, y in line_groups_dict.iteritems():
			if (k != x) and (len(v.intersection(y)) > 0):
				sys.exit(0)

def merge_small_line_groups(line_groups_dict, contour_id_to_data_dict):
	line_group_sizes = {}
	line_group_sizes_list = []
	new_line_groups_dict = {}
	ret_line_groups_dict = {}

	for k, v in line_groups_dict.iteritems():
		line_group_size = get_line_group_size(v, contour_id_to_data_dict)
		line_group_sizes[k] = line_group_size
		line_group_sizes_list.append(line_group_size)

	threshold_line_group_size = max(line_group_sizes_list) * THRESH_LINE_GROUP_SIZE_MULT
	for k, v in line_group_sizes.iteritems():
		if v >= threshold_line_group_size:
			new_line_groups_dict[k] = copy.deepcopy(line_groups_dict[k])

	for k, v in new_line_groups_dict.iteritems():
		ret_line_groups_dict[k] = copy.deepcopy(v)

	for k, v in line_group_sizes.iteritems():
		if v < threshold_line_group_size:
			closest_line_g_id = get_closest_line_group_id(k, new_line_groups_dict, line_groups_dict, contour_id_to_data_dict)
			if closest_line_g_id is None:
				ret_line_groups_dict[k] = copy.deepcopy(line_groups_dict[k])
			else:
				ret_line_groups_dict[closest_line_g_id] = (ret_line_groups_dict[closest_line_g_id]).union(line_groups_dict[k])

	return ret_line_groups_dict

def get_closest_line_group_id(group_id_to_merge, candidate_line_groups_dict, line_groups_dict, contour_id_to_data_dict):
	min_dist = None
	closest_g_id = None

	for k, v in candidate_line_groups_dict.iteritems():
		dist = get_dist_between_contour_groups_vert(line_groups_dict[group_id_to_merge], v, contour_id_to_data_dict)
		if (min_dist is None) or (dist < min_dist):
			min_dist = dist
			closest_g_id = k

	return closest_g_id

def get_dist_between_contour_groups_vert(g_1, g_2, contour_id_to_data_dict):
	"""
	Gets vertical distance between two contour groups	
	"""

	current_group = g_1
	new_group = g_2
	centroid_pair_distances = []
	for x in current_group:
		for y in new_group:
			dist = abs(contour_id_to_data_dict[x]["centroid"][1] - contour_id_to_data_dict[y]["centroid"][1])
			centroid_pair_distances.append(dist)

	return min(centroid_pair_distances)

def get_line_group_size(contour_ids_list, contour_id_to_data_dict):
	size = float(0)
	for c_id in contour_ids_list:
		size += contour_id_to_data_dict[c_id]["area"]

	return size

def filter_groups_of_contours_dict(groups_of_contours_dict, contour_id_to_data_dict):
	"""
	For purposes of trying to speed things up, remove groups of contours
	that have horizontal overlap
	"""
	groups_of_contours_dict_filtered = {}
	for k, v in groups_of_contours_dict.iteritems():
		overlaps_with_existing = False
		for x, y in groups_of_contours_dict_filtered.iteritems():
			if groups_of_contours_overlaps(v, y, contour_id_to_data_dict):
				overlaps_with_existing = True
				break

		if not overlaps_with_existing:
			groups_of_contours_dict_filtered[k] = v

	return groups_of_contours_dict_filtered

def groups_of_contours_overlaps(g_1, g_2, contour_id_to_data_dict):
	current_group = g_1
	new_group = g_2
	centroid_pair_distances = []
	for x in current_group:
		for y in new_group:
			dist = euclid_dist(contour_id_to_data_dict[x]["centroid"], contour_id_to_data_dict[y]["centroid"])
			centroid_pair_distances.append((x, y, dist))

	centroid_pair_distances.sort(key=lambda t: t[2])

	current_group_c_id = centroid_pair_distances[0][0]
	new_group_c_id = centroid_pair_distances[0][1]

	x_1, y_1, w_1, h_1 = contour_id_to_data_dict[current_group_c_id]["contour_bounding_b_info"]
	x_2, y_2, w_2, h_2 = contour_id_to_data_dict[new_group_c_id]["contour_bounding_b_info"]

	X_l_g1 = x_1
	Y_l_g1 = y_1
	X_r_g1 = X_l_g1 + w_1
	Y_r_g1 = Y_l_g1 + h_1

	X_l_g2 = x_2
	Y_l_g2 = y_2
	X_r_g2 = X_l_g2 + w_2
	Y_r_g2 = Y_l_g2 + h_2

	# vbd_1 < 0 should be true for us to return True
	# (or vbd_2 < 0 as set below)
	vbd_1 = max(Y_l_g1, Y_l_g2) - min(Y_r_g1, Y_r_g2)

	if vbd_1 < 0:
		return True
	else:
		return False

def debug_line_groups_dict(line_groups_dict, contour_id_to_data_dict, color_img_orig):
	color_for_box_mult = 0.05
	output_img_copy = np.copy(color_img_orig)

	for k, v in line_groups_dict.iteritems():
		for contour_id in v:
			x, y, w, h = contour_id_to_data_dict[contour_id]["contour_bounding_b_info"]
			cv2.rectangle(output_img_copy,(x,y),(x+w,y+h),(0,min(255 * color_for_box_mult, 255),0),2)
		color_for_box_mult += 0.05

	cv2.imshow('debug_line_groups_dict', output_img_copy)
	cv2.waitKey(0)

def debug_line_groups_dict_one_group_at_time(line_groups_dict, contour_id_to_data_dict, 
	color_img_orig):

	for k, v in line_groups_dict.iteritems():
		output_img_copy = np.copy(color_img_orig)
		for contour_id in v:
			x, y, w, h = contour_id_to_data_dict[contour_id]["contour_bounding_b_info"]
			cv2.rectangle(output_img_copy,(x,y),(x+w,y+h),(0,255,0),2)

		cv2.imshow('debug_line_groups_dict', output_img_copy)
		cv2.waitKey(0)


def create_level_3_boxes(contour_id_to_data_dict, CENTROID_DISTANCES_CACHE_DICT):
	"""
	I think the main remaining way to optimize this function is to take the 
	for k, v in filtered_groups_of_contours_dict.iteritems() and multiprocess
	that for loop. Each starting "seed" group can be processed in a separate
	process. We do want to share the data structures groups_used_as_seeds,
	groups_included_in_line_group, and line_groups_dict.
	"""

	"""
	First step - Create a dictionary D of groups of connected contour boxes
	Key is an ID for a group of connected connected contours, the value is
	the set of IDs for the connected contours (where an ID for a connected
	contour is from contour_id_to_data_dict). Only add a group of connected
	contours to the dict if the group is not already present in the dict

	NOTE - Steps 2 and 3 should be using the IDs of the groups of connected contours
	So, a group of groups of connected contours should be represented by as a set of IDs
	for the groups of connected contours

	Second step - For each group G of connected contours in the dictionary D,
	use it as the starting seed. Then do the following:
	1) Loop through D to see what other groups can connected to G. Extend
	G by adding these groups. We now have a group of groups of connected contours.
	2) Repeat 1) until the group of group of connected contours stops changing (
	until we do a loop through D where nothing is added).

	At the end of the second step, we now have groups of groups of connected contours.
	These should correspond to our "lines". Note that a line is NOT just a rectangle
	since a rectangle may be slanted. It's best to think of a line as a group of groups
	of connected contours.

	Third step - Filter out duplicates and subsets from the groups of groups of connected
	contours. Since each group of groups of connected contours is just represented using
	a set of IDs for the groups of connected contours, this should be straightforward - toss out
	a set S if it's a duplicate of a set S' and toss out S if it's a subset of a set S'. 
	"""
	"""
	Remaining to do for this function:
	All is done
	"""

	added_groups = set()
	groups_of_contours_dict = {}

	group_of_conts_id = 1

	for k, v in contour_id_to_data_dict.iteritems():
		group_ids_list = [k]
		for connected_cont_id in v["connected_contours"]:
			group_ids_list.append(connected_cont_id)

		group_ids = frozenset(group_ids_list)
		if group_ids in added_groups:
			continue
		else:
			added_groups.add(group_ids)
			# group_ids is a set of the contur IDs that make up this group
			groups_of_contours_dict[group_of_conts_id] = group_ids
			group_of_conts_id += 1

	# finished building groups_of_contours_dict, ie finished 
	# the "first step" as described in the docstring for this function
	line_groups_dict = {}
	line_group_id = 1


	# getting rid of filtering. it was causing problems because there would be groups
	# with vertical overlap, so one group would get filtered out, but the
	# two groups had a large horizontal gap between them, so both need to be used as seeds
	filtered_groups_of_contours_dict = groups_of_contours_dict
	# filtered_groups_of_contours_dict = filter_groups_of_contours_dict(groups_of_contours_dict, contour_id_to_data_dict)	
	
	groups_used_as_seeds = set()
	groups_included_in_line_group = set()
	contour_ids_included_in_line_group = set()

	for k, v in filtered_groups_of_contours_dict.iteritems():
		# a group only needs to be used as a seed at most once
		# and if it's already been included in a line group, then we don't need to use it as a seed
		# (and also don't need to include it in other line groups for that matter)
		# since a group should appear in exactly 1 line group
		if (k in groups_used_as_seeds) or (k in groups_included_in_line_group):
			continue

		groups_used_as_seeds.add(k)
		groups_included_in_line_group.add(k)

		line_group = set()
		for contour_id in v:
			if contour_id not in contour_ids_included_in_line_group:
				line_group.add(contour_id)
				contour_ids_included_in_line_group.add(contour_id)

		if len(line_group) == 0:
			continue

		line_group_changed = True
		# added_groups is a set of the group IDs that we've extended group k with
		added_groups = set()
		added_groups.add(k)

		while line_group_changed:
			line_group_changed = False
			for x, y in groups_of_contours_dict.iteritems():
				if (x in added_groups) or (x in groups_included_in_line_group):
					continue
				else:
					# y here is the set of contour IDs making up group with id "x"
					if should_extend_with_group_x(line_group, y, contour_id_to_data_dict, CENTROID_DISTANCES_CACHE_DICT):
						added_groups.add(x)
						groups_included_in_line_group.add(x)
						for contour_id in y:
							if contour_id not in contour_ids_included_in_line_group:
								line_group.add(contour_id)
								contour_ids_included_in_line_group.add(contour_id)

						line_group_changed = True

		if len(line_group) > 0:
			line_groups_dict[line_group_id] = line_group
			line_group_id += 1

	# we've now finished building line groups dict
	# keys are just ids for line groups
	# values are just sets of contour ids that make up a line group
	# each line group should hopefully correspond to a line
	# the second step is now done
	# now just need to filter out duplicate line groups or
	# a line group that's a subset of another line group

	line_groups_dict_filtered = filter_line_groups_dict(line_groups_dict)
	return line_groups_dict_filtered

def filter_line_groups_dict(line_groups_dict):
	line_groups_dict_filtered = {}
	added_sets = set()

	for k, v in line_groups_dict.iteritems():
		is_added = False
		for s in added_sets:
			if (v == s) or (v.issubset(s)):
				is_added = True

		if not is_added:
			line_groups_dict_filtered[k] = v

	return line_groups_dict_filtered

def should_extend_with_group_x(current_group, new_group, contour_id_to_data_dict, CENTROID_DISTANCES_CACHE_DICT):
	"""
	current_group is a set of contour ids
	new_group is a set of contour ids
	contour_id_to_data_dict is the dict by the same name in get_blocks

	This function returns True if current_group can be extended with new_group
	and False otherwise
	"""

	"""
	Logic for this is going to be similar as for checking whether two contours
	are connected and will be the following:

	Look at the centroids of the bounding boxes that make up each group.
	Let c_1 be the set of centroids for G_1 and c_2 be the set of centroids
	for G_2. Set d = min(c_1*, c_2*) where c_1* in c_1 and c_2* in c_2. So,
	we compute the distance between every centroid c_1* in c_1 and c_2* in c_2
	and take the pair with the minimum distance and define d to be that distance.
	d is intuitively the distance between groups G_1 and G_2. 

	Then define w_1 and h_1 to be the maximum width and height across all widths
	and heights of the bounding boxes that make up G_1, and define w_2 and h_2
	similarly. Define a d_max_g_1 in an analogous way to d_max_c_1 as in the
	function for checking when to connect 2 contours. And define a d_max_g_2.

	Now check that some_min_dist_constant < d < d_max_g_1
	And some_min_dist_constant < d < d_max_g_2
	This intuitively checks that the two groups G_1 and G_2 are reasonably close 
	together

	Finally, we take the bounding boxes that c_1* and c_2* came
	from and check that there's vertical overlap between the two.
	(using the same quantity vbd as used in the function for checking
	if two contours are connected). Actually, we also take
	c_1** and c_2** where these are the "second nearest" pair of
	centroids for which c_1** != c_1* and c_2** != c_2*. If
	the bounding boxes for c_1** and c_2** have vertical overlap,
	then that also counts and is enough. So just one pair needs
	to have vertical overlap.

	So we just do these two checks - centroid distance and vertical overlap.
	That's enough.
	"""
	centroid_pair_distances = []
	for x in current_group:
		for y in new_group:
			x_centroid = contour_id_to_data_dict[x]["centroid"]
			y_centroid = contour_id_to_data_dict[y]["centroid"]
			centroid_pair_key = frozenset((x_centroid, y_centroid))
			if centroid_pair_key in CENTROID_DISTANCES_CACHE_DICT:
				dist = CENTROID_DISTANCES_CACHE_DICT[centroid_pair_key]
			else:	
				dist = euclid_dist(x_centroid, y_centroid)
				CENTROID_DISTANCES_CACHE_DICT[centroid_pair_key] = dist

			centroid_pair_distances.append((x, y, dist))

	centroid_pair_distances.sort(key=lambda t: t[2])
	dist_G1_G2 = centroid_pair_distances[0][2]

	heights_list_current_group = []
	widths_list_current_group = []

	heights_list_new_group = []
	widths_list_new_group = []

	for g in current_group:
		heights_list_current_group.append(contour_id_to_data_dict[g]["contour_bounding_b_info"][3])
		widths_list_current_group.append(contour_id_to_data_dict[g]["contour_bounding_b_info"][2])
	
	for g in new_group:
		heights_list_new_group.append(contour_id_to_data_dict[g]["contour_bounding_b_info"][3])
		widths_list_new_group.append(contour_id_to_data_dict[g]["contour_bounding_b_info"][2])

	c_group_h = max(heights_list_current_group)
	c_group_w = max(widths_list_current_group)

	n_group_h = max(heights_list_new_group)
	n_group_w = max(widths_list_new_group)

	d_max_c_group = CONSTANT_C_D_GROUPS * max(c_group_h, c_group_w)	
	d_max_n_group = CONSTANT_C_D_GROUPS * max(n_group_h, n_group_w)

	if not (-1 < dist_G1_G2 < d_max_c_group):
		return False

	if not (-1 < dist_G1_G2 < d_max_n_group):
		return False

	current_group_c_id = centroid_pair_distances[0][0]
	new_group_c_id = centroid_pair_distances[0][1]

	x_1, y_1, w_1, h_1 = contour_id_to_data_dict[current_group_c_id]["contour_bounding_b_info"]
	x_2, y_2, w_2, h_2 = contour_id_to_data_dict[new_group_c_id]["contour_bounding_b_info"]

	X_l_g1 = x_1
	Y_l_g1 = y_1
	X_r_g1 = X_l_g1 + w_1
	Y_r_g1 = Y_l_g1 + h_1

	X_l_g2 = x_2
	Y_l_g2 = y_2
	X_r_g2 = X_l_g2 + w_2
	Y_r_g2 = Y_l_g2 + h_2

	# vbd_1 < 0 should be true for us to return True
	# (or vbd_2 < 0 as set below)
	vbd_1 = max(Y_l_g1, Y_l_g2) - min(Y_r_g1, Y_r_g2)

	if vbd_1 < (-0.3 * c_group_h):
		return True
	
	found_flag = False

	for p in centroid_pair_distances:
		if (p[0] != current_group_c_id) and (p[1] != new_group_c_id):
			found_flag = True
			current_group_c_id = p[0]
			new_group_c_id = p[1]
			break

	if found_flag:
		x_1, y_1, w_1, h_1 = contour_id_to_data_dict[current_group_c_id]["contour_bounding_b_info"]
		x_2, y_2, w_2, h_2 = contour_id_to_data_dict[new_group_c_id]["contour_bounding_b_info"]

		X_l_g1 = x_1
		Y_l_g1 = y_1
		X_r_g1 = X_l_g1 + w_1
		Y_r_g1 = Y_l_g1 + h_1

		X_l_g2 = x_2
		Y_l_g2 = y_2
		X_r_g2 = X_l_g2 + w_2
		Y_r_g2 = Y_l_g2 + h_2

		# vbd_1 < 0 should be true for us to return True
		# (or vbd_2 < 0 as set below)
		vbd_2 = max(Y_l_g1, Y_l_g2) - min(Y_r_g1, Y_r_g2)
	else:
		# some token number > 0
		vbd_2 = 100000

	if not ((vbd_1 < (-0.3 * c_group_h)) or (vbd_2 < (-0.3 * c_group_h))):
		return False

	return True

def debug_connections_contour_id_to_data_dict(contour_id_to_data_dict, debug_binary_output_img):
	for k, v in contour_id_to_data_dict.iteritems():
		output_img_copy = np.copy(debug_binary_output_img)
		x, y, w, h = v["contour_bounding_b_info"]

		cv2.rectangle(output_img_copy,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imshow('debug_connections_dict', output_img_copy)
		cv2.waitKey(0)

		for connected_c_id in v["connected_contours"]:
			x, y, w, h = contour_id_to_data_dict[connected_c_id]["contour_bounding_b_info"]
			cv2.rectangle(output_img_copy,(x,y),(x+w,y+h),(0,255,0),2)

		cv2.imshow('debug_connections_dict', output_img_copy)
		cv2.waitKey(0)

def euclid_dist(c_1_centroid, c_2_centroid):
	# this is a very fast implementation of euclid_dist
	# already. as seen in profile_output_linalg_norm
	# using numpy's linalg norm is actually much slower
	x_1 = float(c_1_centroid[0])
	y_1 = float(c_1_centroid[1])

	x_2 = c_2_centroid[0]
	y_2 = c_2_centroid[1]

	dist = math.sqrt(((x_2 - x_1)**2 + (y_2 - y_1)**2))
	return dist

def contour_pair_is_connected(c_1, c_2, CENTROID_DISTANCES_CACHE_DICT):
	c_1_h = c_1[1][3]
	c_1_w = c_1[1][2]
	c_1_centroid = c_1[2]

	c_2_h = c_2[1][3]
	c_2_w = c_2[1][2]
	c_2_centroid = c_2[2]

	d_max_c_1 = CONSTANT_C_D * max(c_1_h, c_1_w)
	d_max_c_2 = CONSTANT_C_D * max(c_2_h, c_2_w)

	centroid_pair_key = frozenset((c_1_centroid, c_2_centroid))

	if centroid_pair_key in CENTROID_DISTANCES_CACHE_DICT:
		euclid_d = CENTROID_DISTANCES_CACHE_DICT[centroid_pair_key]
	else:
		euclid_d = euclid_dist(c_1_centroid, c_2_centroid)
		CENTROID_DISTANCES_CACHE_DICT[centroid_pair_key] = euclid_d

	if not (CONSTANT_D_MIN < euclid_d < d_max_c_1):
		return False

	if not (CONSTANT_D_MIN < euclid_d < d_max_c_2):
		return False

	max_p_size = max(c_1[3], c_2[3])
	min_p_size = min(c_1[3], c_2[3])

	if not (float(max_p_size) / min_p_size < T_PSR):
		return False

	# below notation has the denotations as in
	# pg 266 of Text Blocking 1.pdf

	X_l_c1 = c_1[1][0]
	Y_l_c1 = c_1[1][1]
	X_r_c1 = c_1[1][0] + c_1_w
	Y_r_c1 = c_1[1][1] + c_1_h

	X_l_c2 = c_2[1][0]
	Y_l_c2 = c_2[1][1]
	X_r_c2 = X_l_c2 + c_2_w
	Y_r_c2 = Y_l_c2 + c_2_h

	hbd = max(X_l_c1, X_l_c2) - min(X_r_c1, X_r_c2)
	vbd = max(Y_l_c1, Y_l_c2) - min(Y_r_c1, Y_r_c2)
	
	if (hbd >= 0) and (vbd >= 0):
		return False

	# we require vertical overlap between the bounding
	# boxes, which is different from the paper, because
	# we want text line blocks in the next stage, not just
	# general grouping blocks
	# I've tested both with the below condition and without it
	# having the below condition (equivalent to requiring
	# vertical overlap) is MUCH more useful than not having it
	# with the below, I get only text blocks that are "roughly"
	# horizontal and the text blocks basically correspond to what I want
	# without any false positive inclusions that don't have vertical overlap
	if not (vbd < (-0.2 * ((c_1_h + c_2_h) / 2))):
		return False

	return True

def get_text_blocks_list(binary_output_img, color_img_orig, contour_bounding_boxes_list,
	CENTROID_DISTANCES_CACHE_DICT):
	"""
	This function is done and it works very well (have visualized
	the results with the now commented out line debug_connections(....) below)
	"""
	for c in contour_bounding_boxes_list:
		connected_contours = []
		for c_prime in contour_bounding_boxes_list:
			if contour_pair_is_connected(c, c_prime, CENTROID_DISTANCES_CACHE_DICT):
				connected_contours.append(c_prime)

		c.append(connected_contours)

	"""
	for each contour in contour_bounding_boxes_list,
	we've now computed what other contours it's
	connected to and appended the list of its
	connections to the list representing the contour's
	info
	the below function is a debugging function
	to make sure that the above code worked by
	visualizing the connections for a given contour
	"""
	debug_color_output_img = np.copy(color_img_orig)
	# debug_connections(contour_bounding_boxes_list, debug_color_output_img)

def debug_connections(contour_bounding_boxes_list, debug_binary_output_img):
	cv2.imshow('debug_connections_1', debug_binary_output_img)
	cv2.waitKey(0)

	for c in contour_bounding_boxes_list:
		output_img_copy = np.copy(debug_binary_output_img)
		x, y, w, h = c[1]
		cv2.rectangle(output_img_copy,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imshow('debug_connections_2', output_img_copy)
		cv2.waitKey(0)

		for connected_c in c[5]:
			x, y, w, h = connected_c[1]
			cv2.rectangle(output_img_copy,(x,y),(x+w,y+h),(0,255,0),2)

		cv2.imshow('debug_connections_3', output_img_copy)
		cv2.waitKey(0)

def get_contour_bounding_boxes_list(binary_output_img, color_img):
	"""
	This function has been tested and works.

	Return value:

	List of tuples of contours in the binary_output_img and associated info.
	Format is:

	[(contour points, bounding box info, centroid coordinates, area (of the contour, not the bounding box)), 
	(contour points, ...), ...]	
	"""


	# invert img since contours need to be in white, bg in black
	binary_output_img = cv2.bitwise_not(binary_output_img)
	# cv2.imshow("binary_output_img", binary_output_img)
	# cv2.waitKey(0)

	kernel = np.ones((2, 2), np.uint8)


	binary_output_img = cv2.dilate(binary_output_img, kernel, iterations=1)
	# cv2.imshow("dilated binary_output_img", binary_output_img)
	# cv2.waitKey(0)

	img_for_contours = binary_output_img


	# RETR_EXTERNAL works correctly and is what I want
	# ie, only the external contours, not nested ones	
	_, contours, _ = cv2.findContours(img_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	
	height, width, channels = color_img.shape
	img_area = float(height) * width


	"""
	below list is a list of tuples created by the line
	contour_bounding_boxes_list.append(
		... (see the line in the for loop below)
	)
	"""
	contour_bounding_boxes_list = []

	contour_id = 1

	color_img_orig = np.copy(color_img)


	for c in contours:
		area = int(cv2.contourArea(c))
		color_img = np.copy(color_img_orig)
		
		if (area < (0.10 * img_area)) and (area > 20):
			cv2.drawContours(color_img, [c], -1, (0, 255, 0), thickness=cv2.FILLED)
			x,y,w,h = cv2.boundingRect(c)
			bounding_box_area = float(w) * h

			# below is the check ii. dens(CC_i) < T_dens
			# in Text Blocking 1.pdf
			if (float(area) / bounding_box_area) < 0.08:
				continue

			centroid = (int(x + float(w)/2), int(y + float(h)/2))

			# generally using the order x, y for
			# coordinate elements in the below appended tuple

			contour_bounding_boxes_list.append(
				[c, (x, y, w, h), centroid, area, contour_id]
			)

			contour_id += 1

			"""
			cv2.circle(color_img, centroid, 10, (0, 255, 0), 2)
			cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.imshow('contour_points_img bounding box', color_img)
			cv2.waitKey(0)
			"""

	return contour_bounding_boxes_list