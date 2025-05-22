import os
import cv2
import time
import re
from google import genai
from google.genai import types

# Inicializa o cliente com a chave de API
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Controle de chamadas à API
gemini_cache = {}
gemini_calls = 0
last_reset_time = 0
daily_gemini_calls = 0
DAILY_LIMIT = 1500
MAX_PER_MINUTE = 15

def limpar_placa(texto):
    """Remove qualquer caractere que não seja letra ou número."""
    return re.sub(r'[^A-Z0-9]', '', texto.upper())

def can_call_gemini():
    global gemini_calls, last_reset_time
    current_time = time.time()
    if current_time - last_reset_time >= 60:
        gemini_calls = 0
        last_reset_time = current_time
    if gemini_calls < MAX_PER_MINUTE:
        gemini_calls += 1
        return True
    return False

def can_call_gemini_daily():
    global daily_gemini_calls
    if daily_gemini_calls < DAILY_LIMIT:
        daily_gemini_calls += 1
        return True
    return False

def gemini_fallback(img_bgr, is_new_plate, modelo="gemini-2.0-flash"):
    """
    Usa o Gemini para tentar reconhecer a placa. Retorna a placa corrigida ou None.
    """
    img_hash = hash(img_bgr.tobytes())
    if img_hash in gemini_cache:
        return gemini_cache[img_hash]

    if not (can_call_gemini() and can_call_gemini_daily()):
        print("[AVISO] Limite de chamadas à API do Gemini atingido.")
        return None

    # Codifica a imagem em bytes
    _, buf = cv2.imencode(".jpg", img_bgr)
    img_bytes = buf.tobytes()
    image_part = types.Part.from_bytes(img_bytes)

    prompt = (
        "Você é um leitor de placas brasileiro. "
        "Retorne SOMENTE a placa no formato AAA9999 ou AAA9A99. "
        "Se não encontrar nenhuma placa válida, responda 'NONE'."
    )

    try:
        response = client.models.generate_content(
            model=modelo,
            contents=[image_part, prompt],
            safety_settings={"HARASSMENT": "block_none"}
        )
    except Exception as e:
        print(f"[ERRO] Falha na chamada do Gemini: {e}")
        gemini_cache[img_hash] = None
        return None

    texto = response.text.strip().upper()
    if texto == "NONE":
        gemini_cache[img_hash] = None
        return None

    texto = limpar_placa(texto)
    gemini_cache[img_hash] = texto
    return texto
