import cv2
import HandTrackingModule as htm
import numpy as np
import streamlit as st


def main():
    if 'start' not in st.session_state:
        st.session_state.start = False
    if 'stop' not in st.session_state:
        st.session_state.stop = False

    st.title("Sign Language Detection - OPENCV")

    frame_placeholder = st.empty()
    col1, col2, col3, col4 = st.columns(4)

    with col2:
        st.button("Start", on_click=toggle_start)
    with col3:
        st.button("Stop", on_click=toggle_stop)

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

            img, handType, landmarks = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)

            if len(lmList) != 0:
                points = np.asarray([lmList[0], lmList[5], lmList[17]])
                normal_vector = np.cross(
                    points[2] - points[0], points[1] - points[2])
                normal_vector = normal_vector//np.linalg.norm(normal_vector)

                fingers_up = detector.fingersUp()

                # For Hello

                if (handType == "Right" and fingers_up[0] == 1 and fingers_up[1]==1 and fingers_up[2]==1 and fingers_up[3]==1 and np.array_equal(normal_vector, [-1.0, 0.0, -1.0])) or \
                        (handType == "Left" and fingers_up[0] == 0 and fingers_up[1]==1 and fingers_up[2]==1 and fingers_up[3]==1 and np.array_equal(normal_vector, [0.0, 0.0, 0.0])):
                    cv2.rectangle(img,(450,20),(550,60),(255,255,255),-1)
                    cv2.putText(img, "Hello", (460, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                # For Yes

                if (handType == "Right" and fingers_up[0] == 0 and fingers_up[4]==1 and np.array_equal(normal_vector, [-1.0, 0.0, -1.0])) or \
                        (handType == "Left" and fingers_up[0] == 1 and fingers_up[4]==1 and np.array_equal(normal_vector, [0.0, 0.0, 0.0])):
                    cv2.rectangle(img,(450,20),(530,60),(255,255,255),-1)
                    cv2.putText(img, "Yes", (460, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                # For Please

                if (handType == "Right" and fingers_up[0] == 1 and fingers_up[1] == 1 and np.array_equal(normal_vector, [0.0, 0.0, -1.0])) or \
                        (handType == "Left" and fingers_up[0] == 0 and fingers_up[1] == 1 and np.array_equal(normal_vector, [-1.0, 0.0, 0.0])):
                    cv2.rectangle(img,(450,20),(570,60),(255,255,255),-1)
                    cv2.putText(img, "Please", (460, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                    
                    
                # For Nice

                if (handType == "Right" and fingers_up[0] == 1 and fingers_up[1] == 0 and fingers_up[2]==1 and np.array_equal(normal_vector, [-1.0, 0.0, -1.0])) or \
                        (handType == "Left" and fingers_up[0] == 0 and fingers_up[1] == 0 and fingers_up[2]==1 and np.array_equal(normal_vector, [0.0, 0.0, 0.0])):
                    cv2.rectangle(img,(450,20),(540,60),(255,255,255),-1)
                    cv2.putText(img, "Nice", (460, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                # For Like

                if (handType == "Right" and fingers_up[1]==0  and np.array_equal(normal_vector, [0.0, 0.0, -1.0])) or \
                (handType == "Left" and fingers_up[1]==0 and np.array_equal(normal_vector, [-1.0, 0.0, 0.0])) :
                    cv2.rectangle(img,(450,20),(530,60),(255,255,255),-1)
                    cv2.putText(img, "Like", (460, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                    
                    
                # For Dislike

                if (handType == "Right" and fingers_up[1]==0  and np.array_equal(normal_vector, [-1.0, -1.0, -1.0])) or \
                (handType == "Left" and fingers_up[1]==0 and np.array_equal(normal_vector, [0.0, -1.0, 0.0])) :
                    cv2.rectangle(img,(450,20),(570,60),(255,255,255),-1)
                    cv2.putText(img, "Dislike", (460, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                # For Thank You

                if handType == "Both" and fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 1 and fingers_up[4] == 1:
                    cv2.rectangle(img,(450,20),(630,60),(255,255,255),-1)
                    cv2.putText(img, "Thank You", (460, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

            else:
                cv2.rectangle(img,(300,20),(630,60),(255,255,255),-1)
                cv2.putText(img, "No hands Detected", (310, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
