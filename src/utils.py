import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from collections import Counter, defaultdict
# from .gemini_fallback import gemini_fallback, can_call_gemini

ocr = PaddleOCR(
        use_angle_cls=True, 
        lang='en', 
        show_log=False
    )


def change_char_in_position(word, position):
    if position < len(word):
        if word[position].isdigit():
            digit = word[position]
            if digit == '8':
                word = word[:position] + 'B' + word[position+1:]
            elif digit == '1':
                word = word[:position] + 'I' + word[position+1:]
            elif digit == '0':
                word = word[:position] + 'O' + word[position+1:]
            elif digit == '5':
                word = word[:position] + 'S' + word[position+1:]
            elif digit == '6':
                word = word[:position] + 'G' + word[position+1:]
            elif digit == '3':
                word = word[:position] + 'J' + word[position+1:]
    return word

def detect_blue_strip(image):
    if image is None:
        return False
    
    height, width = image.shape[:2]
    top_strip = image[0:int(height * 0.25), 0:width]  # faixa de cima

    hsv = cv2.cvtColor(top_strip, cv2.COLOR_BGR2HSV)

    # Faixa de azul (com tolerância maior para placas borradas)
    lower_blue = np.array([110, 160, 65])
    upper_blue = np.array([130, 255, 255])
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


def extract_plate_from_image(image_path, plates):
    # Preprocess the image
    is_new_plate = detect_blue_strip(image_path)
    result = ocr.ocr(image_path, cls=True)
        
    if not result or result[0] is None:
        print("Nenhum resultado encontrado.")
        return None    
    
    detected_words = []
    for line in result:
        for word_info in line:
            text = word_info[1][0].replace(" ", "")
            text = limpar_placa(text)
            print(f"Texto detectado: {text}")
            if text.lower() == "brasil":
                is_new_plate = True
            else:
                detected_words.append((text, word_info[1][1]))  # (text, confidence)
                    
    # print(f"Palavras detectadas: {detected_words}")
    if is_new_plate:
        print(f"É nova placa? {is_new_plate}")    
    
    # Tenta cada palavra isolada
    for word_tuple in detected_words:
        corrected = correct_plate(word_tuple[0], is_new_plate)
        if corrected and corrected not in plates:
            return corrected

    # Tenta combinações de palavras
    for i in range(len(detected_words)):
        for j in range(i+1, len(detected_words)):
            combined = detected_words[i][0] + detected_words[j][0]
            print(f"Combinando: {combined}")
            corrected = correct_plate(combined, is_new_plate)
            if corrected:
                return corrected    
    
    # print("Fazendo fallback para Gemini...")
    # if can_call_gemini():
    #     gemini_plate = gemini_fallback(image_path if isinstance(image_path, np.ndarray) else cv2.imread(image_path), is_new_plate)
    #     if gemini_plate:
    #         return gemini_plate
    # else:
    #     print("Limite de chamadas à API do Gemini atingido. Aguardando...")

    return None


def hamming_distance(a, b):
    if len(a) != len(b):
        return float('inf')
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

def agrupar_placas_por_hamming_completo(plate_counts, max_dist=1):
    placas = list(plate_counts.keys())
    grafo = defaultdict(list)

    for i in range(len(placas)):
        for j in range(i + 1, len(placas)):
            if hamming_distance(placas[i], placas[j]) <= max_dist:
                grafo[placas[i]].append(placas[j])
                grafo[placas[j]].append(placas[i])

    visitado = set()
    grupos = []
    

    def dfs(placa, grupo):
        visitado.add(placa)
        grupo.append(placa)
        for vizinho in grafo[placa]:
            if vizinho not in visitado:
                dfs(vizinho, grupo)

    for placa in placas:
        if placa not in visitado:
            grupo = []
            dfs(placa, grupo)
            grupos.append(grupo)

    return grupos

