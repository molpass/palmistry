import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
from pillow_heif import register_heif_opener

# 결과 이미지에 한국어 라벨이 깨지지 않도록 한글 폰트를 설정한다(설치된 첫 폰트 사용).
matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def heic_to_jpeg(heic_dir, jpeg_dir):
    register_heif_opener()  
    image = Image.open(heic_dir)
    image.save(jpeg_dir, "JPEG")

def remove_background(jpeg_dir, path_to_clean_image):
    if jpeg_dir[-4:] in ['heic', 'HEIC']:
        heic_to_jpeg(jpeg_dir, jpeg_dir[:-4] + 'jpg')
        jpeg_dir = jpeg_dir[:-4] + 'jpg'
    img = cv2.imread(jpeg_dir)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 80], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(result)  
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)
    img[mask == 255] = 255
    cv2.imwrite(path_to_clean_image, img)

def resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value):
    pil_img = Image.open(path_to_warped_image)
    pil_img_clean = Image.open(path_to_warped_image_clean)
    pil_img.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_mini)
    pil_img_clean.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_clean_mini)

def save_result(im, contents, resize_value, path_to_result):
    if im is None:
        print_error()
    else:
        heart_content_1, heart_content_2, head_content_1, head_content_2, life_content_1, life_content_2 = contents
        image_height, image_width = im.size
        fontsize = 12
        
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            left=False,         # ticks along the top edge are off
            labelbottom=False,
            labelleft=False
        )

        note_1 = '* 참고: 이 프로그램은 그저 재미를 위한 것입니다! 결과는 가벼운 마음으로 받아들여 주세요.'
        note_2 = '   손금에 대해 더 알아보고 싶다면 https://www.allure.com/story/palm-reading-guide-hand-lines 를 추천합니다'

        plt.title(' 손금 분석 결과를 확인하세요!', fontsize=14, y=1.01)

        plt.text(image_width + 15, 15, "<감정선>", color='r', fontsize=fontsize)
        plt.text(image_width + 15, 35, heart_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 55, heart_content_2, fontsize=fontsize)
        plt.text(image_width + 15, 80, "<두뇌선>", color='g', fontsize=fontsize)
        plt.text(image_width + 15, 100, head_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 120, head_content_2, fontsize=fontsize)
        plt.text(image_width + 15, 145, "<생명선>", color='b', fontsize=fontsize)
        plt.text(image_width + 15, 165, life_content_1, fontsize=fontsize)
        plt.text(image_width + 15, 185, life_content_2, fontsize=fontsize)

        plt.text(image_width + 15, 230, note_1, fontsize=fontsize-1, color='gray')
        plt.text(image_width + 15, 250, note_2, fontsize=fontsize-1, color='gray')

        plt.imshow(im)
        plt.savefig(path_to_result, bbox_inches = "tight")

def print_error():
    print('손금 선이 제대로 검출되지 않았습니다! 다른 손바닥 이미지를 사용해 주세요.')