from PIL import Image
import os

def load_image_if_exists(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} は存在しません。")
    img = Image.open(filepath)
    return img

def test_load_image_file_check():
    try:
        img = load_image_if_exists("test.png")  # 存在する画像ファイルに変更してください
        print("ファイル存在チェックOK")
    except FileNotFoundError as e:
        print("ファイル存在チェックNG:", e)

test_load_image_file_check()
