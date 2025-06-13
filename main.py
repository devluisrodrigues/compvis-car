from src.utils import extract_plate_from_image, agrupar_placas_por_hamming_completo
from src.models import YOLO, classify_and_crop
import cv2 as cv
from collections import Counter

# VIDEO_PATH and FPS
VIDEO_PATH = 'video1.MOV'
FPS = 30

def main():
    max_dist = 2

    capture = cv.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Could not open video file")
        return

    success, input = capture.read()
    if not success:
        print("Could not read first frame")
        return

    model = YOLO('last.pt')
    cv.namedWindow(VIDEO_PATH)
    delay = int(1000 / FPS)

    frame_count = 0
    all_plates = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_count += 1

        if frame_count % (FPS/3) == 0:
            frame_count = 0
            cropped_imgs = classify_and_crop(frame, model)

            for img in cropped_imgs:
                print("Processing image...")
                plate = extract_plate_from_image(img, [])
                if plate:
                    print(f"Plate detected: {plate}")
                    all_plates.append(plate)
                else:
                    print("No valid plate detected.")
                print('-' * 20)
                
                
        frame = cv.resize(frame, (640, 480))

        cv.imshow(VIDEO_PATH, frame)

        key = cv.waitKey(delay)
        if key == ord('q') or not cv.getWindowProperty(VIDEO_PATH, cv.WND_PROP_VISIBLE):
            print("Exiting...\n")
            break

    # --- Estatísticas finais ---
    print(f"Total raw plate reads: {len(all_plates)}\n")

    plate_counts = Counter(all_plates)
    print("Unique plates sorted by frequency:\n")
    for i, (plate, count) in enumerate(plate_counts.most_common(), start=1):
        print(f"{i}. {plate}: {count}x")

    grupos = agrupar_placas_por_hamming_completo(plate_counts, max_dist)
    
    placas_finais = []
    
    print(f"\nMost likely plates by group (based on Hamming distance ≤ {max_dist}):\n")
    for i, grupo in enumerate(grupos, 1):
        # Seleciona a placa com maior frequência no grupo
        placa_mais_frequente = max(grupo, key=lambda p: plate_counts[p])
        placas_finais.append(placa_mais_frequente)
        total = sum(plate_counts[placa] for placa in grupo)
        if total > 1:
            print(f"{i}. {placa_mais_frequente} (group: {' | '.join(grupo)}) --> total: {total}x")


    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
