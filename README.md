# writing-line-segmentation
Processes images that contain text / writing and groups characters into lines. This is an adapted + extended version of the blocking algorithm presented in Text Binarization in Color Documents by Badekas, Nikolaou, and Papamarkos.

![An example result from running the line segmentation code on test_images/PR8.png.tiff. Note that this
only shows one of the line segments found from PR8](https://github.com/davidxc/writing-line-segmentation/blob/master/line_segment_images/line_segment_example_result_PR8.png)

The above is an example result from running the line segmentation code on test_images/PR8.png.tiff. Note that this only shows one of the line segments found from PR8. The full result also marks each of the other writing lines in the image as an individual line segment (only showing 1 line segment in the above example image so as to be clear on what an individual line segment looks like).

The code requires the libraries cv2 and numpy. run_text_blocking.py has an example of how to use the line blocking function. The line blocking function is the "get_blocks" function in text_blocking.py, and this is the function you should import and use.
