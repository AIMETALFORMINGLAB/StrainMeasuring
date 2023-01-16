import cv2
import numpy as np
import pandas as pd
import imutils
from imutils import perspective
from imutils import contours

# 트랙커 객체 생성자 함수 리스트 ---①
trackers = [cv2.TrackerCSRT_create, cv2.TrackerTLD_create]
trackers2 = [cv2.TrackerCSRT_create, cv2.TrackerTLD_create]
trackerIdx = 0  # 트랙커 생성자 함수 선택 인덱스
trackerIdx2 = 0  # 트랙커 생성자 함수 선택 인덱스
tracker = None
isFirst = True

def onChange(x):
    pass

    
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
#video_src = 0 # 비디오 파일과 카메라 선택 ---②
video_src = 'Vbending.avi'
cap = cv2.VideoCapture(video_src)
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)
win_name = 'Tracking APIs'
j=0
scaler = 1

f=open('Tensile test3 tracking_result.txt', 'w')
f2=open('Tensile test3 tracking_result2.txt', 'w')


while cap.isOpened():
    ret, frame = cap.read()
    ret, frame2 = cap.read()
    ret, frame3 = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))
    frame2 = cv2.resize(frame2, (int(frame2.shape[1] * scaler), int(frame2.shape[0] * scaler)))
    frame3 = cv2.resize(frame3, (int(frame3.shape[1] * scaler), int(frame3.shape[0] * scaler)))
    if not ret:
        print('Cannot read video file')
        break
    img_draw = frame.copy()

    if tracker is None: # 트랙커 생성 안된 경우
        cv2.putText(img_draw, "Press the Space to set ROI!!", \
            (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    else:
        ok, bbox = tracker.update(frame)   # 새로운 프레임에서 추적 위치 찾기 ---③
        ok2, bbox2=tracker2.update(frame) 
        (x,y,w,h) = bbox
        (x2,y2,w2,h2)=bbox2
        timer = cv2.getTickCount()
        ps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        if ok: # 추적 성공
            cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), \
                          (0,255,0), 2, 1)
            cv2.rectangle(img_draw, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), \
                          (0,255,0), 2, 1)

            #1번째 원 추적                          
            mask = np.zeros_like(frame2)
            cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (255,255,255), -1)  
            frame4 = cv2.bitwise_and(frame2, mask)          
            frame4gray = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
            
#            cv2.namedWindow("binary detection")
#            cv2.createTrackbar("binary low", "binary detection", 0,255, onChange)
#            cv2.setTrackbarPos("binary low", "binary detection", 127)

#            while cv2.waitKey(1) != ord('q'):
    
#               bin_low = cv2.getTrackbarPos("binary low", "binary detection")

#               _, gray2 = cv2.threshold(frame4gray, bin_low, 255, cv2.THRESH_BINARY )
#               cv2.imshow('binary detection',gray2)
    

#           cv2.destroyAllWindows()  
            
#            roi=frame4gray[int(y):int(y+h), int(x):int(x+w)]
            rows, cols = frame4gray.shape[:2]
            mask = np.zeros((rows+2, cols+2), np.uint8)
            newVal = (0,0,0)
            seed = (1,1)
            loDiff, upDiff = (10,10,10), (10,10,10)

           
            ret, frame4bin = cv2.threshold(frame4gray, 45, 255, cv2.THRESH_BINARY_INV )
            retval = cv2.floodFill(frame4bin, mask, seed, newVal, loDiff, upDiff)
            #cv2.imshow('Canny Edge', frame4bin)
            #cv2.waitKey(0)
            edged = cv2.Canny(frame4bin, 10, 100)
            #frame4gray = cv2.threshold(frame4gray,127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #cv2.imshow('win_name', frame4gray)
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5)) #히스토그램균등화-흑백만적용
            #frame4gray=clahe.apply(frame4gray)
            circles = cv2.HoughCircles(frame4bin,cv2.HOUGH_GRADIENT,1.5,30, param1=80,param2=30,minRadius=0,maxRadius=0)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:  
                try:          
                    orig = frame.copy()
                    # 모멘트 계산 
                    mmt = cv2.moments(c)
                    cx = int(mmt['m10']/mmt['m00'])
                    cx1 = float(mmt['m10']/mmt['m00'])
                    cy = int(mmt['m01']/mmt['m00'])
                    cy1 = float(mmt['m01']/mmt['m00'])
                    cv2.circle(img_draw, (cx, cy), 5, (0, 0, 255), -1)
                    print(j,cx1, cy1, file=f)   
                except ZeroDivisionError:
                    print("ZeroDivision")                    
            #cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
            
            #2번째 원 추적
            mask = np.zeros_like(frame3)
            cv2.rectangle(mask, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), (255,255,255), -1)  
            frame5 = cv2.bitwise_and(frame3, mask)    
            
            frame5gray = cv2.cvtColor(frame5, cv2.COLOR_BGR2GRAY)
            rows, cols = frame5gray.shape[:2]
            mask = np.zeros((rows+2, cols+2), np.uint8)
            newVal = (0,0,0)
            seed = (1,1)
            loDiff, upDiff = (10,10,10), (10,10,10)           
            ret, frame5bin = cv2.threshold(frame5gray, 45, 255, cv2.THRESH_BINARY_INV )
            retval = cv2.floodFill(frame5bin, mask, seed, newVal, loDiff, upDiff)          
            edged2 = cv2.Canny(frame5bin, 10, 100)            


            cnts2 = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
            cnts2 = imutils.grab_contours(cnts2)
            for c in cnts2:  
                try:          
                    orig = frame.copy()
                    # 모멘트 계산 
                    mmt = cv2.moments(c)
                    cx = int(mmt['m10']/mmt['m00'])
                    cx1 = float(mmt['m10']/mmt['m00'])
                    cy = int(mmt['m01']/mmt['m00'])
                    cy1 = float(mmt['m01']/mmt['m00'])
                    cv2.circle(img_draw, (cx, cy), 5, (0, 0, 255), -1)
                    print(j,cx1, cy1, file=f2)   
                except ZeroDivisionError:
                    print("ZeroDivision")      
     
            circles2 = cv2.HoughCircles(frame5gray,cv2.HOUGH_GRADIENT,1,30, param1=80,param2=30,minRadius=0,maxRadius=20)
            
            j=j+1            
            time=j/30
            

                    

            #print(j,int(x+w/2), int(y+h/2),int(x2+w2/2), int(y2+h2/w),file=f)        
            cv2.putText(img_draw, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);    
        else : # 추적 실패
            cv2.putText(img_draw, "Tracking fail.", (100,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    trackerName = tracker.__class__.__name__
    cv2.putText(img_draw, str(trackerIdx) + ":"+trackerName , (100,20), \
                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2,cv2.LINE_AA)

    cv2.imshow(win_name, img_draw)


    key = cv2.waitKey(delay) & 0xff
    # 스페이스 바 또는 비디오 파일 최초 실행 ---④
    if key == ord(' ') or (video_src != 0 and isFirst): 
        isFirst = False
        roi = cv2.selectROI(win_name, frame, False)  # 초기 객체 위치 설정
        roi2 = cv2.selectROI(win_name, frame, False)  # 초기 객체 위치 설정
        
        if roi[2] and roi[3]:         # 위치 설정 값 있는 경우
            tracker = trackers[trackerIdx]()    #트랙커 객체 생성 ---⑤
            isInit = tracker.init(frame, roi)
        
        if roi2[2] and roi2[3]:         # 위치 설정 값 있는 경우
            tracker2 = trackers2[trackerIdx]()    #트랙커 객체 생성 ---⑤
            isInit2 = tracker2.init(frame, roi2)
        
    elif key in range(48, 56): # 0~7 숫자 입력   ---⑥
        trackerIdx = key-48     # 선택한 숫자로 트랙커 인덱스 수정
        if bbox is not None:
            tracker = trackers[trackerIdx]() # 선택한 숫자의 트랙커 객체 생성 ---⑦
            isInit = tracker.init(frame, bbox) # 이전 추적 위치로 추적 위치 초기화
    elif key == 27 : 
        break
else:
    print( "Could not open video")
cap.release()
cv2.destroyAllWindows()