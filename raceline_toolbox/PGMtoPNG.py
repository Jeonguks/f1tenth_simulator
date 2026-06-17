import tkinter as tk
from tkinter import filedialog
from PIL import Image

# PGM 파일을 선택하고 PNG로 저장하는 함수
def convert_pgm_to_png():
    # Tkinter 창을 숨기기
    root = tk.Tk()
    root.withdraw()

    # 파일 열기 대화상자 열기 (PGM 파일 선택)
    pgm_file = filedialog.askopenfilename(title="PGM 파일 선택", filetypes=[("PGM files", "*.pgm")])
    
    if not pgm_file:
        print("파일이 선택되지 않았습니다.")
        return

    # 저장 대화상자 열기 (PNG 파일 저장 위치 선택)
    png_file = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], title="PNG 파일 저장")

    if not png_file:
        print("저장 경로가 선택되지 않았습니다.")
        return

    # PGM 파일을 열고 PNG로 저장
    with Image.open(pgm_file) as img:
        img.save(png_file, 'PNG')
    
    print(f"{pgm_file}이 {png_file}로 변환되었습니다.")

# 실행
convert_pgm_to_png()
