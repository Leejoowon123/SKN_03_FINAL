import subprocess
import sys
import os

# 현재 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# main.py 경로
main_dir = os.path.abspath(os.path.join(current_dir, ".."))
if main_dir not in sys.path:
    sys.path.append(main_dir)

from DeepFM import DeepFM
from Preprocessing_Process import Processing
from recommendation import Recommender
import config



class Musical_Process:
    def execute_script(self, script_name):
        # 현재 파일 기준으로 스크립트 경로 설정
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        # 가상환경의 Python 실행 경로 가져오기
        venv_python = sys.executable

        # print(f"Executing {script_path} with {venv_python}...")
        try:
            subprocess.run([venv_python, script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing {script_name}: {e}")
            raise

if __name__ == "__main__":
    final_processing_path = f'{config.file_path}/{config.last_processing_file_name}'
    deepfm_model_path = f'{config.model_file_path}/{config.model_name}'

    if not os.path.exists(final_processing_path):
        # print("Preprocessing_Process 실행")
        Musical_Process.execute_script("Preprocessing_Process.py")
    else:
        # print("Preprocessing_Process 생략")
        pass

    # DeepFM.py 실행 조건
    if not os.path.exists(deepfm_model_path):
        # print("DeepFM 실행")
        Musical_Process.execute_script("DeepFM.py")
    else:
        pass
        # print("DeepFM 생략")

