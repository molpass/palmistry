from PIL import Image, ImageDraw
import cv2
import mediapipe as mp

def measure(path_to_warped_image_mini, lines):
    heart_thres_x = 0
    head_thres_x = 0
    life_thres_y = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(path_to_warped_image_mini), 1)
        image_height, image_width, _ = image.shape

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand_landmarks = results.multi_hand_landmarks[0]

        zero = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
        one = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
        five = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
        nine = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
        thirteen = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

        heart_thres_x = image_width * (1 - (nine + (five - nine) * 2 / 5))
        head_thres_x = image_width * (1 - (thirteen + (nine - thirteen) / 3))
        life_thres_y = image_height * (one + (zero - one) / 3)

    im = Image.open(path_to_warped_image_mini)
    width = 3
    if (None in lines) or (len(lines) < 3):
        return None, None
    else:
        draw = ImageDraw.Draw(im)

        heart_line = lines[0]
        head_line = lines[1]
        life_line = lines[2]

        heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
        heart_line_tip = heart_line_points[0]
        heart_content_1 = '감정선은 연애, 우정, 헌신 등 마음에 관한 모든 일을 관장합니다.'
        if heart_line_tip[0] < heart_thres_x:
            heart_content_2 = '감정선이 깁니다. 사랑하거나 아끼는 사람과 오랜 관계를 맺게 됨을 의미합니다.'
        else:
            heart_content_2 = '감정선이 짧습니다. 평생 다양한 사람을 만나며 폭넓은 관계를 맺게 됨을 의미합니다.'
        draw.line(heart_line_points, fill="red", width=width)

        head_line_points = [tuple(reversed(l[:2])) for l in head_line]
        head_line_tip = head_line_points[-1]
        head_content_1 = '두뇌선은 지적 호기심과 탐구에 대해 알려줍니다.'
        if head_line_tip[0] > head_thres_x:
            head_content_2 = '두뇌선이 깁니다. 평생 폭넓은 주제를 탐구하게 됨을 의미합니다.'
        else:
            head_content_2 = '두뇌선이 짧습니다. 한 가지 주제에 매료되어 깊이 파고들게 됨을 의미합니다.'
        draw.line(head_line_points, fill="green", width=width)

        life_line_points = [tuple(reversed(l[:2])) for l in life_line]
        life_line_tip = life_line_points[-1]
        life_content_1 = '생명선은 당신의 경험, 활력, 열정을 드러냅니다. 주의하세요 — 수명의 길이와는 전혀 관련이 없습니다!'
        if life_line_tip[1] > life_thres_y:
            life_content_2 = '생명선이 깁니다. 혼자보다는 다른 사람과 함께 문제를 해결하는 경향을 의미합니다.'
        else:
            life_content_2 = '생명선이 짧습니다. 독립적이고 자율적임을 의미합니다.'
        draw.line(life_line_points, fill="blue", width=width)

        # draw.line([(heart_thres_x, 0), (heart_thres_x, image_height)], fill="red")
        # draw.line([(head_thres_x, 0), (head_thres_x, image_height)], fill="green")
        # draw.line([(0, life_thres_y), (image_width, life_thres_y)], fill="blue")

        contents = [heart_content_1, heart_content_2, head_content_1, head_content_2, life_content_1, life_content_2]
        return im, contents