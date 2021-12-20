import cv2
import numpy as np
import matplotlib.pyplot as plt
import bleedfacedetector as fd
import time

model = 'assets/emotion-ferplus-8.onnx'
net = cv2.dnn.readNetFromONNX(model)

def init_emotion(model="assets/emotion-ferplus-8.onnx"):
    global net,emotions
    emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
    net = cv2.dnn.readNetFromONNX(model)

def emotion(image, returndata=False, confidence=0.3):
    
    img_copy = image.copy()
    faces = fd.ssd_detect(img_copy,conf=confidence)
    padding = 3 

    for x,y,w,h in faces:
        face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding]
        gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray, (64, 64))
        processed_face = resized_face.reshape(1,1,64,64)
        net.setInput(processed_face)
        
        Output = net.forward()
        expanded = np.exp(Output - np.max(Output))
        probablities =  expanded / expanded.sum()
        
        prob = np.squeeze(probablities)
        
        predicted_emotion = emotions[prob.argmax()]
        cv2.putText(img_copy,'{}'.format(predicted_emotion),(x,y+h+(1*20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        # Draw rectangular box on detected face
        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
    
    if  returndata:
        return img_copy
    else:
        plt.figure(figsize=(10,10))
        plt.imshow(img_copy[:,:,::-1]);plt.axis("off");

def gen_frames():
    cap = cv2.VideoCapture(0)
    init_emotion()
    fps=0
    init_emotion()
    while(cap.isOpened()):    
        start_time = time.time()
        success,img=cap.read() 

        if success==True:    
            image = cv2.flip(img,1)
            img=cv2.resize(img,(0,0),fx=0.5,fy=0.5)

            image = emotion(image, returndata=True, confidence = 0.8)

            cv2.putText(image, 'FPS: {:.2f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 20, 55), 1)
            cv2.imshow("Emotion Recognition",image)
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            k = cv2.waitKey(1)
            fps= (1.0 / (time.time() - start_time))

            if k == ord('q'):
                break
        else:
            break    
    cap.release()  
cv2.destroyAllWindows() 