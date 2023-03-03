import mediapipe 
import cv2
me_hands=mediapipe.solutions.hands
draw=mediapipe.solutions.drawing_utils
hands=me_hands.Hands(max_num_hands=3,min_detection_confidence=0.7)
cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res=hands.process(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# clr=cv2.circle(conv_img,center=(150,180),radius=200,color=(200,0,0),thickness=4)
    cv2.circle(img,(40,410),30,(255,255,255),cv2.FILLED)
    # cv2.rectangle(img,(20,350),(90,440),(0,255,0),thickness=3)
    tipid=[4,8,12,16,20]
    lmlist=[]
    if res.multi_hand_landmarks :
        for handlms in res.multi_hand_landmarks :
            for id,lm in enumerate(handlms.landmark) :
                cx=lm.x
                cy=lm.y
                lmlist.append([id,cx,cy])
                if len(lmlist)!=0 and len(lmlist)==21 :
                    fing_list=[]

                    #thumb
                    if lmlist[12][1]>lmlist[20][1] :
                        if lmlist[tipid[0]][1]>lmlist[tipid[0]-1][1] :
                            fing_list.append(1)
                        else :
                            fing_list.append(0)
                    else :
                        if lmlist[tipid[0]][1]<lmlist[tipid[0]-1][1] :
                            fing_list.append(1)
                        else :
                            fing_list.append(0)
                    
                    #for other fingers
                    for i in range(1,5) :
                        if lmlist[tipid[i]][2]<lmlist[tipid[i]-2][2]:
                            fing_list.append(1)
                        else :
                            fing_list.append(0)
                    # print(fing_list)

                    if len(fing_list)!=0 :
                        fing_count=fing_list.count(1)
                
                    cv2.putText(img,str(fing_count),(25,425),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),5)
                draw.draw_landmarks(img,handlms,me_hands.HAND_CONNECTIONS,draw.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=1),draw.DrawingSpec(color=(0,0,0),thickness=2))

    cv2.imshow('Finger',img)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break
cv2.destroyAllWindows()
print(lmlist)