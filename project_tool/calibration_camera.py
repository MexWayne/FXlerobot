import cv2

#cam_w = cv2.VideoCapture(4)  # /dev/video4
cam_r = cv2.VideoCapture(0)  # /dev/video2
cam_l = cv2.VideoCapture(2)  # /dev/video0


# 强制降低分辨率和帧率
for cam in [cam_r, cam_l]:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 15)
    # 尝试设置 MJPG（如果摄像头支持，会大幅降低带宽）
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("Right camera opened:", cam_r.isOpened())
print("Left  camera opened:", cam_l.isOpened())


while(True):

    if False:
        ret_w, frame_w = cam_w.read()  # ret==True/False: read successfully or not; frame: image
        if not ret_w:
            print("Failed to read the image from wrist camera.")
            break
        # display image
        cv2.imshow('Video_from_wrist', frame_w)
        # press ESC key to exit
        key_w = cv2.waitKey(1)
        if key_w == 27:
            break

    ####################################################################################################
    if True:
        ret_r, frame_r = cam_r.read()  # ret==True/False: read successfully or not; frame: image
        if not ret_r:
            print("Failed to read the image from right camera.")
            break


        ####### draw the cross ########
        # get the center 
        height, width = frame_r.shape[:2]
        center_x, center_y = width // 2, height // 2
        # the length of the cross 
        line_length = 20
        color = (255, 0, 0)  # 红色 (B, G, R)
        thickness = 3
        # horizon line 
        cv2.line(frame_r,
                 (center_x, center_y - line_length),
                 (center_x, center_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_r,
                 (center_x - line_length, center_y),
                 (center_x + line_length, center_y),
                 color, thickness)
        


        # get the point0
        height, width = frame_r.shape[:2]
        point0_x, point0_y = width // 2 - 105, height // 2 + 47
        # the length of the cross 
        line_length = 20
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 1
        # horizon line 
        cv2.line(frame_r,
                 (point0_x, point0_y - line_length),
                 (point0_x, point0_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_r,
                 (point0_x - line_length, point0_y),
                 (point0_x + line_length, point0_y),
                 color, thickness)
        # create the text
        cv2.putText(frame_r,
                    text="point0", org=(point0_x - 40, point0_y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255), 
                    thickness=1)


        

        # get the point1
        height, width = frame_r.shape[:2]
        point1_x, point1_y = width // 2 + 63, height // 2 - 149
        # the length of the cross 
        line_length = 20
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 1
        # horizon line 
        cv2.line(frame_r,
                 (point1_x, point1_y - line_length),
                 (point1_x, point1_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_r,
                 (point1_x - line_length, point1_y),
                 (point1_x + line_length, point1_y),
                 color, thickness)
        # create the text
        cv2.putText(frame_r,
                    text="point1", org=(point1_x, point1_y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255), 
                    thickness=1)        


        

        # get the point2
        height, width = frame_r.shape[:2]
        point2_x, point2_y = width // 2 + 220, height // 2 + 12
        # the length of the cross 
        line_length = 20
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 1
        # horizon line 
        cv2.line(frame_r,
                 (point2_x, point2_y - line_length),
                 (point2_x, point2_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_r,
                 (point2_x - line_length, point2_y),
                 (point2_x + line_length, point2_y),
                 color, thickness)
        cv2.putText(frame_r,
                    text="point2", org=(point2_x + 20, point2_y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255), 
                    thickness=1)   



        # get the point3
        height, width = frame_r.shape[:2]
        point3_x, point3_y = width // 2 + 11, height // 2 + 202
        # the length of the cross 
        line_length = 20
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 1
        # horizon line 
        cv2.line(frame_r,
                 (point3_x, point3_y - line_length),
                 (point3_x, point3_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_r,
                 (point3_x - line_length, point3_y),
                 (point3_x + line_length, point3_y),
                 color, thickness)

        

        # draw area
        line_length = 20
        color = (0, 255, 0)  # 红色 (B, G, R)
        thickness = 1
        # line 01
        cv2.line(frame_r, (point0_x, point0_y), (point1_x, point1_y), color, thickness)
        # line 12
        cv2.line(frame_r, (point1_x, point1_y), (point2_x, point2_y), color, thickness)
        # line 23
        cv2.line(frame_r, (point2_x, point2_y), (point3_x, point3_y), color, thickness)
        # line 30
        cv2.line(frame_r, (point3_x, point3_y), (point0_x, point0_y), color, thickness)


        #######################

        # display image
        cv2.imshow('Video_from_right', frame_r)
        cv2.moveWindow("Video_from_right", 50, 10)
        # press ESC key to exit
        key_r = cv2.waitKey(1)
        if key_r == 27:
            break
    ####################################################################################################






    ####################################################################################################
    if True:
        ret_l, frame_l = cam_l.read()  # ret==True/False: read successfully or not; frame: image
        if not ret_l:
            print("Failed to read the image from left camera.")
            break


        ####### draw the cross ########
        # get the center 
        height, width = frame_l.shape[:2]
        center_x, center_y = width // 2, height // 2
        # the length of the cross 
        line_length = 20
        color = (255, 0, 0)  # 红色 (B, G, R)
        thickness = 3
        # horizon line 
        cv2.line(frame_l,
                 (center_x, center_y - line_length),
                 (center_x, center_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_l,
                 (center_x - line_length, center_y),
                 (center_x + line_length, center_y),
                 color, thickness)
        


        # get the point0
        height, width = frame_l.shape[:2]
        point0_x, point0_y = width // 2 - 208, height // 2 + 28
        # the length of the cross 
        line_length = 20
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 1
        # horizon line 
        cv2.line(frame_l,
                 (point0_x, point0_y - line_length),
                 (point0_x, point0_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_l,
                 (point0_x - line_length, point0_y),
                 (point0_x + line_length, point0_y),
                 color, thickness)
        # create the text
        cv2.putText(frame_l,
                    text="point0", org=(point0_x - 40, point0_y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255), 
                    thickness=1)


        

        # get the point1
        height, width = frame_l.shape[:2]
        point1_x, point1_y = width // 2 - 55, height // 2 - 141
        # the length of the cross 
        line_length = 20
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 1
        # horizon line 
        cv2.line(frame_l,
                 (point1_x, point1_y - line_length),
                 (point1_x, point1_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_l,
                 (point1_x - line_length, point1_y),
                 (point1_x + line_length, point1_y),
                 color, thickness)
        # create the text
        cv2.putText(frame_l,
                    text="point1", org=(point1_x, point1_y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255), 
                    thickness=1)        
        

        # get the point2
        height, width = frame_l.shape[:2]
        point2_x, point2_y = width // 2 + 125, height // 2 + 45
        # the length of the cross 
        line_length = 20
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 1
        # horizon line 
        cv2.line(frame_l,
                 (point2_x, point2_y - line_length),
                 (point2_x, point2_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_l,
                 (point2_x - line_length, point2_y),
                 (point2_x + line_length, point2_y),
                 color, thickness)
        cv2.putText(frame_l,
                    text="point2", org=(point2_x + 20, point2_y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255), 
                    thickness=1)    



        # get the point3
        height, width = frame_l.shape[:2]
        point3_x, point3_y = width // 2 + 15, height // 2 + 205
        # the length of the cross 
        line_length = 20
        color = (0, 0, 255)  # 红色 (B, G, R)
        thickness = 1
        # horizon line 
        cv2.line(frame_l,
                 (point3_x, point3_y - line_length),
                 (point3_x, point3_y + line_length),
                 color, thickness)
        # vertical line 
        cv2.line(frame_l,
                 (point3_x - line_length, point3_y),
                 (point3_x + line_length, point3_y),
                 color, thickness)
        

        # draw area
        line_length = 20
        color = (0, 255, 0)  # 红色 (B, G, R)
        thickness = 1
        # line 01
        cv2.line(frame_l, (point0_x, point0_y), (point1_x, point1_y), color, thickness)
        # line 12
        cv2.line(frame_l, (point1_x, point1_y), (point2_x, point2_y), color, thickness)
        # line 23
        cv2.line(frame_l, (point2_x, point2_y), (point3_x, point3_y), color, thickness)
        # line 30
        cv2.line(frame_l, (point3_x, point3_y), (point0_x, point0_y), color, thickness)
        #######################

        # display image
        cv2.imshow('Video_from_left', frame_l)
        cv2.moveWindow("Video_from_left", 50, 600)
        # press ESC key to exit
        key_l = cv2.waitKey(1)
        if key_l == 27:
            break
    ####################################################################################################




#cam_w.release()
cam_l.release()
cam_r.release()
