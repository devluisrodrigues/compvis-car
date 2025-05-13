import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

def change_char_in_position(word, position):
    if position < len(word):
        if word[position].isdigit():
            digit = word[position]
            if digit == '8':
                word = word[:position] + 'B' + word[position+1:]
            elif digit == '1':
                word = word[:position] + 'I' + word[position+1:]
    return word

def detect_blue_strip(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False

    height, width = image.shape[:2]
    top_strip = image[0:int(height * 0.15), 0:width]  # faixa de cima

    hsv = cv2.cvtColor(top_strip, cv2.COLOR_BGR2HSV)

    # Faixa de azul (com tolerância maior para placas borradas)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    blue_ratio = cv2.countNonZero(mask) / (top_strip.shape[0] * top_strip.shape[1])
    return blue_ratio > 0.03

def limpar_placa(texto):
    """Remove qualquer caractere que não seja letra ou número."""
    return re.sub(r'[^A-Z0-9]', '', texto.upper())

def correct_plate(word, is_new_plate):
    word = word.replace("-", "").replace(".", "").replace(" ", "")
    word = limpar_placa(word)

    if len(word) < 7:
        return None

    for pos in [0, 1, 2]:
        if word[pos].isdigit():
            word = change_char_in_position(word, pos)
    
    if is_new_plate and len(word) > 4 and word[4].isdigit():
        word = change_char_in_position(word, 4)

    # Verifica os dois formatos válidos
    if re.match(r'^[A-Z]{3}\d{4}$', word) or re.match(r'^[A-Z]{3}\d[A-Z]\d{2}$', word):
        return word

    return None

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def extract_plate_from_image(image_path):
    is_new_plate = detect_blue_strip(image_path)
    result = ocr.ocr(image_path, cls=True)
    
    if not result or result[0] is None:
        print("No text detected.")
        return None    
    
    detected_words = []
    for line in result:
        for word_info in line:
            text = word_info[1][0].replace(" ", "")
            if text.lower() == "brasil":
                is_new_plate = True
            else:
                detected_words.append((text, word_info[1][1]))  # (text, confidence)
    
    print(f"Palavras detectadas: {detected_words}")
    print(f"É nova placa? {is_new_plate}")    
    
    # Tenta cada palavra isolada
    for word_tuple in detected_words:
        corrected = correct_plate(word_tuple[0], is_new_plate)
        if corrected:
            return corrected

    # Tenta pares de palavras combinadas
    for i in range(len(detected_words)):
        for j in range(i+1, len(detected_words)):
            combined = detected_words[i][0] + detected_words[j][0]
            corrected = correct_plate(combined, is_new_plate)
            if corrected:
                return corrected

    return None
