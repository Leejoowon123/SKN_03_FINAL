import pandas as pd
import glob
import os
from datetime import datetime


"""일별 예매율을 계산 함수"""
def calculate_daily_booking_rate(sales, ticket_price, max_capacity):
    max_possible_sales = ticket_price * max_capacity
    booking_rate = (sales / max_possible_sales) * 100
    return round(booking_rate, 2)

"""뮤지컬 데이터를 통합해 처리하는 함수"""
def process_musical_data(musical_data):
    """
    Args: musical_data (DataFrame): Musical_Data.csv 데이터
    """
    # 결과 저장할 리스트
    integrated_data_list = []
    
    # Data 폴더에 모든 Casting_Board.csv 파일
    casting_files = glob.glob('Data/*_Casting_Board.csv')
    
    for file_path in casting_files:
        # 파일 이름에서 뮤지컬 제목 추출
        musical_id = os.path.basename(file_path).replace('_Casting_Board.csv', '')
        
        # 해당 뮤지컬 정보
        musical_info = musical_data[musical_data['뮤지컬 제목'] == musical_id].iloc[0]
        
        # 캐스팅 보드 파일 읽기
        casting_data = pd.read_csv(file_path)
        
        for _, row in casting_data.iterrows():
            # 관람일에서 날짜와 요일 분리
            date_parts = row['관람일'].split('(')
            date = date_parts[0]
            day = date_parts[1].replace(')', '')
            
            # 출연진 문자열 생성
            cast_members = ','.join([str(row[f'역할명{i}']) for i in range(1, 6)])
            
            # 일별 예매율 계산
            daily_booking_rate = calculate_daily_booking_rate(
                row['판매액'],
                musical_info['티켓 가격'],
                musical_info['공연장 최대 수용 수']
            )
            
            # 새로운 데이터 행
            new_row = {
                '뮤지컬 제목': musical_id,
                '관람일': date,
                '관람요일': day,
                '관람 시간': row['시간'],
                '티켓 가격': musical_info['티켓 가격'],
                '판매액': row['판매액'],
                '일별 예매율': daily_booking_rate,
                '출연진': cast_members,
                '공연 시설명': musical_info['공연 시설명'],
                '공연장 최대 수용 수': musical_info['공연장 최대 수용 수'],
                '줄거리': musical_info['줄거리'],
                '공연 장르': musical_info['공연 장르명']
            }
            
            integrated_data_list.append(new_row)
    
    integrated_musical_data = pd.DataFrame(integrated_data_list)
    
    return integrated_musical_data

def main():
    """
    메인 함수
    """
    # 결과 파일 경로
    output_dir = 'Data/Final'
    output_file = os.path.join(output_dir, 'Combined_Musical_Data.csv')
    
    # 파일이 이미 존재하는지 확인
    if os.path.exists(output_file):
        print(f"'{output_file}' 파일 이미 존재")
        # 기존 파일 읽어서 반환
        return pd.read_csv(output_file)
    
    # 디렉터리 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 없는 경우 새로 생성
    print(f"'{output_file}' 파일 생성중")
    
    try:
        # Musical_Data.csv 파일
        musical_data = pd.read_csv('Data/Musical_Data.csv')
        
        # 통합 데이터 생성
        integrated_data = process_musical_data(musical_data)
        
        # CSV 파일 저장
        integrated_data.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"'{output_file}' 파일 생성 완료")
        return integrated_data
        
    except FileNotFoundError as e:
        print(f"오류: 필요 데이터 파일 찾을 수 없음. {str(e)}")
        return None
    except Exception as e:
        print(f"오류: {str(e)}")
        return None

if __name__ == "__main__":
    integrated_data = main()