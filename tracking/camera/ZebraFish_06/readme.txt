Each line of an annotation txt file is structured as follows:
frame: The video frame which the annotation is associated with 
id: Identity of the fish
3d_x: x coordinate of 3D head position in world coordinates
3d_y: y coordinate of 3D head position in world coordinates
3d_z: z coordinate of 3D head position in world coordinates
camT_x: x coordinate of 2D head position in the top view in image coordinates
camT_y: y coordinate of 2D head position in the top view in image coordinates
camT_left: x coordinate of the top left corner of the bounding box in the top view in image coordinates
camT_top: y coordinate of the top left corner of the bounding box in the top view in image coordinates
camT_width: Width of the bounding box in the top view in image coordinates
camT_width: Height of the bounding box in the top view in image coordinates
camT_occlusion: Boolean indicating the fish is part of an occlusion in the top view
camF_x: x coordinate of 2D head position in the front view in image coordinates
camF_y: y coordinate of 2D head position in the front view in image coordinates
camF_left: x coordinate of the top left corner of the bounding box in the front view in image coordinates
camF_top: y coordinate of the top left corner of the bounding box in the front view in image coordinates
camF_width: Width of the bounding box in the front view in image coordinates
camF_width: Height of the bounding box in the front view in image coordinates
camF_occlusion: Boolean indicating the fish is part of an occlusion in the front view

Four example lines from a txt file:
1, 1, 19.61, 28.313, 7.93, 1523, 1283, 1377, 1214, 156, 78, 1, 1666, 961, 1438, 911, 251, 85, 0
1, 2, 18.317, 28.636, 8.911, 1464, 1284, 1449, 1241, 139, 61, 1, 1589, 1027, 1559, 968, 215, 99, 0
2, 1, 19.685, 28.348, 7.886, 1527, 1285, 1382, 1217, 155, 76, 1, 1670, 959, 1439, 908, 255, 86, 0
2, 2, 18.197, 28.625, 8.868, 1460, 1284, 1446, 1249, 143, 53, 1, 1580, 1024, 1550, 972, 228, 91, 0


Submission format:

You can submit your tracking result where each row of your submission file has to contain the following values. The values are defined as in the annotation file, and any other values will be ignored.
frame, id, 3d_x, 3d_y, 3d_z
