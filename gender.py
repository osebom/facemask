#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:53:14 2021

@author: ojonyeagwu
"""

import face_recognition
import cv2

webcam_video_stream =cv2.VideoCapture(0)#to have it analyze a video, we change "0" to the path of the video instead

all_face_locations=[] 

while True:
    ret,current_frame= webcam_video_stream
    current_frame_small= cv2.resize(current_frame, (0,0), fx= .25, fy=0.25)#2nd argument is designated size but here we aren't changing it
    #3rd argument re-scales the image (scale factors)
    
    all_face_locations= face_recognition.face_locations(current_frame_small, model="hog")
    
    for index,current_face_location in enumerate(all_face_locations):
        top_pos,right_pos,bottom_pos,left_pos= current_face_location
        
        
        top_pos= top_pos*4
        right_pos= right_pos*4
        left_pos= left_pos*4
        bottom_pos= bottom_pos*4
        
        print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index+1,top_pos, right_pos, bottom_pos, left_pos))
    
        current_face_image= current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
        
        
        AGE_GENDER_MODEL_MEAN_VALUES= (78.426, 87.769, 114.896)
        #In the this tuple, the first value is mean vlaue of the channels of the main image, then height and width
        
        #Create BLOB for current_face_image
        #1st arg is image to convert into blob, 2nd is scale factor which is 1 and means scalling is 100%
        #3rd is size of the blob, 4th is the mean values,
        current_face_image_blob= cv2.nn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB= False)
        
        #GENDER PREDICTION
        
        gender_label_list = ['Male', 'Female']
        gender_protext= ['/Users/ojonyeagwu/Desktop/code/age/deploy_gender.prototxt']
        gender_caffemodel= ['/Users/ojonyeagwu/Desktop/code/age/gender_net.caffemodel']
        
        #Create model from files and provide blob as input
        
        gender_cov_net= cv2. dnn.readNet(gender_caffemodel, gender_protext) #gender convolutional neural.bet
        gender_cov_net.setInput(current_face_image_blob) #Now the model is ready and we're giving it an input
        
        #method 'foward' makes model work, now we have to push that input through nn model
        gender_predictions= gender_cov_net.forward()
        gender= gender_label_list[gender_predictions[0].argmax()]
        
        #AGE PREDICTION
        
        age_label_list = ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)', '(15-20)', '(38-43)', '(48-53)', '(60-(']
        age_protext= ['/Users/ojonyeagwu/Desktop/code/age/deploy_age.prototxt']
        age_caffemodel= ['/Users/ojonyeagwu/Desktop/code/age/age_net.caffemodel']
        
        age_cov_net= cv2. dnn.readNet(gender_caffemodel, gender_protext) #gender convolutional neural.bet
        age_cov_net.setInput(current_face_image_blob) #Now the model is ready and we're giving it an input
        
        #method 'foward' makes model work, now we have to push that input through nn model
        age_predictions= gender_cov_net.forward()
        age= gender_label_list[gender_predictions[0].argmax()]
      
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos+20), (0,0, 255), 2) #the last two arguments are color and thickness
        #first argument is image
        
    font= cv2.FONT_HERSHEY_COMPLEX()
    cv2.putText(current_frame, gender+" "+age+" "+"yrs", (left_pos, bottom_pos), font, 0.5, (0, 255,0), 1)
        

    
    cv2.imshow("Webcam Video", current_frame) #Common title
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #cv2waitKey detects what key is pressed 
        break

#release  the stream and cam
#close all open cv windows       
webcam_video_stream.release()
cv2.destroyAllWindows()