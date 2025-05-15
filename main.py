from src.utils import extract_plate_from_image, agrupar_placas_por_hamming_completo
from src.models import YOLO, classify_and_crop
import cv2 as cv
from collections import Counter

TITLE = 'video'
FPS = 30




def main():
    max_dist = 2

    capture = cv.VideoCapture('video/video1.MOV')
    if not capture.isOpened():
        print("Could not open video file")
        return

    success, input = capture.read()
    if not success:
        print("Could not read first frame")
        return

    model = YOLO('last.pt')
    cv.namedWindow(TITLE)
    delay = int(1000 / FPS)

    frame_count = 0
    all_plates = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_count += 1

        if frame_count % 10 == 0:
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

        cv.imshow(TITLE, frame)

        key = cv.waitKey(delay)
        if key == ord('q') or not cv.getWindowProperty(TITLE, cv.WND_PROP_VISIBLE):
            print("Exiting...\n")
            break

    # --- Estatísticas finais ---
    print(f"Total raw plate reads: {len(all_plates)}\n")

    plate_counts = Counter(all_plates)
    print("Unique plates sorted by frequency:\n")
    for i, (plate, count) in enumerate(plate_counts.most_common(), start=1):
        print(f"{i}. {plate}: {count}x")

    grupos = agrupar_placas_por_hamming_completo(plate_counts, max_dist)

    gabarito = ['SWI3B77', 'CUY1B80', 'GDY4G44', 'STK3G35', 'GCZ7D94', 'CGN9J94', 'SVG1E26', 'SUW1C12', 'FWR5B86', 'DSV7C13', 'SSZ5I11', 'FNC1F27', 'CUH4G26',
                'CUT4H69', 'FIO2H66', 'DGL6088', 'EUD7280', 'TDP8C43', 'DFR7733', 'GEJ4J73', 'GDT2B64', 'DES5518', 'GCK4H60', 'GAR4G31', 'QXF1E08', 'GHV7564',
                'FMJ6399', 'ALF8F22', 'FRY2I96', 'BSX6F35', 'BQU2B91']
    
    placas_finais = []
    
    print(f"\nMost likely plates by group (based on Hamming distance ≤ {max_dist}):\n")
    for i, grupo in enumerate(grupos, 1):
        # Seleciona a placa com maior frequência no grupo
        placa_mais_frequente = max(grupo, key=lambda p: plate_counts[p])
        placas_finais.append(placa_mais_frequente)
        total = sum(plate_counts[placa] for placa in grupo)
        if total > 1:
            print(f"{i}. {placa_mais_frequente} (grupo: {' | '.join(grupo)}) --> total: {total}x")

    print("\nPlates not in the reference list:\n")
    acerto = 0

    for placa in placas_finais:
        if placa not in gabarito:
            print(f"Placa não encontrada: {placa}")
        else:
            acerto += 1    

    print(f"\nTotal de acertos: {acerto} de {len(placas_finais)} placas.\n")
    print(f"Porcentual de acertos: {round((acerto / len(placas_finais)) * 100, 2)}%.\n")
    


    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
