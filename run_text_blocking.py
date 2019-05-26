from text_blocking import get_blocks
import cv2
import numpy as np

def run_text_blocking():
	test_images_to_run = ["PR8.png.tiff"]

	for f in test_images_to_run:
		full_path = "test_images/%s" % f
		binary_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
		color_img = cv2.imread(full_path, cv2.IMREAD_COLOR)

		contour_id_to_data_dict, line_groups_dict = get_blocks(binary_img, color_img, f)

		debug_line_groups_dict_one_group_at_time(line_groups_dict, contour_id_to_data_dict, color_img)

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

if __name__ == "__main__":
	run_text_blocking()