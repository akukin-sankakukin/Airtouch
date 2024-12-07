import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# MediaPipeの設定
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# カメラ入力
cap = cv2.VideoCapture(0)

# 時間記録変数
pinch_start_time = None
dragging = False
click_text_timer = None
pinky_bend_time = None  # 小指の曲げTime
copy_text_timer = None  # コピー表示Time

# 画面の解像度を取得
screen_width, screen_height = pyautogui.size()

# ウィンドウ表示の設定
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

# フィルタリング
previous_index_x, previous_index_y = 0, 0
smooth_factor = 0.7  # 平滑化の係数

def is_pointing_up(hand_landmarks):
    """人差し指が上を向いているかどうかを判定"""
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    if (index_finger_tip.y < index_finger_dip.y < index_finger_pip.y < index_finger_mcp.y and
        middle_finger_tip.y > index_finger_tip.y and
        ring_finger_tip.y > index_finger_tip.y and
        pinky_tip.y > index_finger_tip.y):
        return True
    return False

def is_pointing_down(hand_landmarks):
    """人差し指が下を向いているかどうかを判定"""
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    if (index_finger_tip.y > index_finger_dip.y > index_finger_pip.y > index_finger_mcp.y and
        middle_finger_tip.y < index_finger_tip.y and
        ring_finger_tip.y < index_finger_tip.y and
        pinky_tip.y < index_finger_tip.y):
        return True
    return False

def is_good_gesture(hand_landmarks):
    """goodのジェスチャー判定"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    if (thumb_tip.y < thumb_ip.y < thumb_mcp.y and
        index_finger_tip.y > thumb_mcp.y and
        middle_finger_tip.y > thumb_mcp.y and
        ring_finger_tip.y > thumb_mcp.y and
        pinky_tip.y > thumb_mcp.y):
        return True
    return False

def is_pinky_bent(hand_landmarks):
    """小指の曲がりを判定"""
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    
    # 小指が曲がっている（先端が第2関節よりも下）場合にTrueを返す
    return pinky_tip.y > pinky_dip.y

def display_text(frame, text, duration=1, position=(50, 150)):
    """テキストを表示する関数"""
    global copy_text_timer
    if copy_text_timer is None or time.time() - copy_text_timer > duration:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        copy_text_timer = time.time()


#開始点=========================================================
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #指
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            frame_height, frame_width, _ = frame.shape

            #画面座標
            screen_index_x = np.interp(index_finger_tip.x, [0, 1], [0, screen_width])
            screen_index_y = np.interp(index_finger_tip.y, [0, 1], [0, screen_height])

            #平滑化 
            smooth_index_x = smooth_factor * previous_index_x + (1 - smooth_factor) * screen_index_x
            smooth_index_y = smooth_factor * previous_index_y + (1 - smooth_factor) * screen_index_y
            # ドラッグ中のマウス移動
            if dragging:
                pyautogui.moveTo(smooth_index_x, smooth_index_y)

            if dragging or is_pointing_up(hand_landmarks):
                pyautogui.moveTo(smooth_index_x, smooth_index_y)

            if is_pointing_up(hand_landmarks):
                cv2.putText(frame, "MouseOperation", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            previous_index_x, previous_index_y = smooth_index_x, smooth_index_y

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5

            # クリックかドラッグ
            if distance < 0.09: #1はでかい
                if pinch_start_time is None:
                    pinch_start_time = time.time()
                    
                elif time.time() - pinch_start_time < 0.5 and not dragging:
                    
                    pyautogui.click()
                    display_text(frame, "Click！")
                    pinch_start_time = None
                    time.sleep(0.2)
                    
                elif time.time() - pinch_start_time >= 0.5 and not dragging:
                    #来てない
                    pyautogui.mouseDown()
                    display_text(frame, "Drag")
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()#できない
                    cv2.putText(frame, "Drop！", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    dragging = False
                pinch_start_time = None

            # 円の描画
            if dragging :
                cv2.circle(frame, (int(smooth_index_x), int(smooth_index_y)), 20, (255, 0, 0), 2)  # ドラッグ中の円（緑色）
            if distance < 0.09 and pinch_start_time is not None and time.time() - pinch_start_time < 0.5:
                cv2.circle(frame, (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)), 20, (0, 255, 0), 2)  # クリック時の円（赤色）

            # Good (画面切り替え)
            if is_good_gesture(hand_landmarks):
                cv2.putText(frame, "chenge window", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                pyautogui.hotkey('Alt', 'Tab')  # winを押す
                time.sleep(1)#遅延
                
            # Not good (画面を閉じる)
            if is_pointing_down(hand_landmarks):
                cv2.putText(frame, "close window", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                pyautogui.hotkey('alt', 'f4', 'Fn')  # Alt + F4 を押す
                time.sleep(1)

            # Check for the pinky gesture
           #if is_pinky_bent(hand_landmarks):
            #    if pinky_bend_time is None:
             #       pinky_bend_time = time.time()
              #  elif time.time() - pinky_bend_time < 0.5:  # 曲げから時間が0.5秒以内待機（曲げた状態）
               #     pass
                #else:
                    # 小指を立てた（コピー）            
                 #   pinky_bend_time = None  # タイマーをリセット
           #ここはなしで後で消す
           #できない

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #ブレーク
        break

#終了点=================================================================

cap.release()
cv2.destroyAllWindows()




            
