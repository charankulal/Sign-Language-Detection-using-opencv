import cv2
import time
import HandTrackingModule as htm
import numpy as np
import streamlit as st

def main():
    if 'start' not in st.session_state:
        st.session_state.start = False
    if 'stop' not in st.session_state:
        st.session_state.stop = False
    if 'save' not in st.session_state:
        st.session_state.save = False

    st.title("Sign Language Detection - OPENCV, MediaPipe")

    frame_placeholder = st.empty()
    col1, col2, col3, col4 = st.columns(4)

    with col2:
        st.button("Start", on_click=toggle_start)
    with col3:
        st.button("Stop", on_click=toggle_stop)


    
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()
    handType = "Right"
    normal_vector = np.array([-1.0, 0.0, -1.0])
    
    if st.session_state.start and not st.session_state.stop:
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            if not success:
                break

            img, handType,landmarks = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)

            if len(lmList) != 0:
                # print(lmList[20])
                # print(handType,lmList[4],lmList[8],lmList[12],lmList[16],lmList[20])
                points = np.asarray([lmList[0], lmList[5], lmList[17]])
                normal_vector = np.cross(
                    points[2] - points[0], points[1] - points[2])
                normal_vector = normal_vector//np.linalg.norm(normal_vector)
                
                
                points_no = np.asarray([lmList[4], lmList[8], lmList[20]])
                normal_vector_no = np.cross(
                    points_no[2] - points_no[0], points_no[1] - points[2])
                normal_vector_no = normal_vector_no//np.linalg.norm(normal_vector_no)
                
                # print(handType,normal_vector_no)
                
                # print(handType)
                
                fingers_up=detector.fingersUp()

                # print(handType, fingers_up)
                
                # For Hello

                if (handType == "Right" and fingers_up[0]==1  and np.array_equal(normal_vector, [-1.0, 0.0, -1.0])) or \
                (handType == "Left" and fingers_up[0]==0 and np.array_equal(normal_vector, [0.0, 0.0, 0.0])) :
                # (handType == "Both" and np.array_equal(normal_vector, [0.0, 0.0, 0.0]))    :
                    cv2.putText(img, "Hello", (350, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 1)

                # For Yes
                
                if (handType == "Right" and fingers_up[0]==0  and np.array_equal(normal_vector, [-1.0, 0.0, -1.0])) or \
                (handType == "Left" and fingers_up[0]==1 and np.array_equal(normal_vector, [0.0, 0.0, 0.0])) :
                # (handType == "Both" and np.array_equal(normal_vector, [0.0, 0.0, 0.0]))    :
                    cv2.putText(img, "Yes", (350, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 1)
                    
                # For Please
                
                if (handType == "Right" and fingers_up[0]==1  and np.array_equal(normal_vector, [0.0, 0.0, -1.0])) or \
                (handType == "Left" and fingers_up[0]==0 and np.array_equal(normal_vector, [-1.0, 0.0, 0.0])) :
                # (handType == "Both" and np.array_equal(normal_vector, [0.0, 0.0, 0.0]))    :
                    cv2.putText(img, "Please", (350, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 1)
                    
                    
                # # For No
                        
                # if (handType == "Right" and fingers_up[0]==1  and np.array_equal(normal_vector_no, [-1.0, 0.0, -1.0])) or \
                # (handType == "Left" and fingers_up[0]==0 and np.array_equal(normal_vector_no, [0.0, 0.0, 0.0])) :
                #     cv2.putText(img, "No", (350, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 1)
                    
                # For Thank You
                
                if handType == "Both" and fingers_up[1]==1  and  fingers_up[2]==1 and fingers_up[3]==1  and  fingers_up[4]==1:
                    cv2.putText(img, "Thank You", (350, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 1)
                

            else:
                cv2.putText(img, "No Hand detected", (10, 100),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 1)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            frame_placeholder.image(img, channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def toggle_start():
    st.session_state.start = True
    st.session_state.stop = False

def toggle_stop():
    st.session_state.stop = True
    st.session_state.start = False



if __name__ == "__main__":
    main()
