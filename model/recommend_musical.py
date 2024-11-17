import os
import subprocess
from model.RecommendationModel import MusicalRecommender

""" 파일 존재 여부 확인 → 없으면 Preprocessing.py를 실행하는 함수"""
def check_and_run_preprocessing():

    preprocessed_file_path = 'Data/Final/Combined_Musical_Data.csv'
    
    if not os.path.exists(preprocessed_file_path):
        print("전처리된 데이터 없음. 전처리 시작")
        try:
            # Preprocessing.py 실행
            subprocess.run(['python', 'model/Preprocessing.py'], check=True)
            
            if not os.path.exists(preprocessed_file_path):
                raise Exception("전처리 실패")
                
            print("전처리 완료")
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"전처리 스크립트 실행 중 오류 발생: {str(e)}")
        except Exception as e:
            raise Exception(f"전처리 중 오류 발생: {str(e)}")
    else:
        print("데이터가 이미 존재")

"""사용자 입력 → 뮤지컬 추천 함수"""
def get_musical_recommendations():
    try:
        # 전처리 파일 확인 및 실행
        check_and_run_preprocessing()
        
        # 추천 시스템 초기화
        print("추천 시스템 초기화...")
        recommender = MusicalRecommender()
        
        # 데이터 로드 및 전처리
        print("데이터 로드 및 전처리...")
        recommender.load_and_preprocess_data()
        
        # 모델 생성 및 학습
        print("모델 생성 및 학습...")
        recommender.create_deepfm_model()
        recommender.train_model()
        
        # 모델 성능 평가
        print("\n=== 모델 성능 평가 ===")
        metrics = recommender.evaluate_model()
        print(f"Loss: {metrics['Loss']:.4f}")
        print(f"MAE: {metrics['MAE(%)']:.2f}%")
        print(f"RMSE: {metrics['RMSE(%)']:.2f}%")
        
        # 장르 매핑 정의
        genre_mapping = {
            1: 'Historical',
            2: 'Romance',
            3: 'Drama',
            4: 'Fantasy',
            5: 'Comedy'
        }
        
        # 사용자 입력 받기
        print("\n=== 뮤지컬 추천 시스템 ===")
        favorite_actor = input("\n선호하는 배우: ")
        
        # 장르 선택 메뉴 출력
        print("\n선호하는 장르를 선택")
        for num, genre in genre_mapping.items():
            print(f"{num}. {genre}")
        
        while True:
            try:
                genre_choice = int(input("번호 입력(1-5): "))
                if genre_choice in genre_mapping:
                    favorite_genre = genre_mapping[genre_choice]
                    break
                else:
                    print("1부터 5까지의 숫자만 입력해주세요.")
            except ValueError:
                print("숫자만 입력해주세요.")
        
        # 입력값 검증
        if not favorite_actor.strip():
            raise ValueError("배우 이름:")
        
        # 추천 받기
        print("\n추천 뮤지컬을 찾는 중...")
        recommendations = recommender.recommend_musicals(favorite_actor, favorite_genre)
        
        # 추천 결과 출력
        print("\n=== 추천 뮤지컬 목록 ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}번째 추천 뮤지컬")
            print(f"제목: {rec['뮤지컬 제목']}")
            print(f"예측 예매율: {rec['예측 예매율']:.2f}%")
            print(f"공연일: {rec['관람일']} ({rec['관람요일']})")
            print(f"장르: {rec['공연 장르']}")
            print(f"공연장: {rec['공연 시설명']}")
            print(f"티켓 가격: {rec['티켓 가격']:,}원")
            print(f"출연진: {rec['출연진']}")
            print("-" * 50)
            
        return recommendations
            
    except ValueError as ve:
        print(f"\n입력 오류: {str(ve)}")
        return None
    except Exception as e:
        print(f"\n시스템 오류: {str(e)}")
        return None

"""메인 함수"""
def main():

    while True:
        recommendations = get_musical_recommendations()
        
        if recommendations:
            # 계속할지 묻기
            choice = input("\n다른 추천? (y/n): ")
            if choice.lower() != 'y':
                print("\n추천 시스템을 종료")
                break
        else:
            # 오류 발생 시 다시 시도 여부 확인
            choice = input("\n다시 시도? (y/n): ")
            if choice.lower() != 'y':
                print("\n추천 시스템 종료")
                break
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 