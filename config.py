"""파일 경로"""
file_path = "C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final"
processed_performance_details = "processed_performance_details.json"
add_genre_file_name = "add_genre_story.json"
df_with_negatives_path = 'C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final/df_with_negatives.json'
picture_file_path = 'C:/SKN_3_MyProject/SKN_03_FINAL/READMEImages/Performance.jpg'
save_model_path = "C:/SKN_3_MyProject/SKN_03_FINAL/Data/Model/Recommend.pkl"



# genre
unique_genres = [
    '드라마/감동', '음악중심/주크박스', '판타지/어드벤처', '코미디/유머', 
    '코미디/유머, 음악중심/주크박스', '드라마/감동, 음악중심/주크박스',
    '드라마/감동, 코미디/유머', '코미디/유머, 액션/스릴러', '액션/스릴러',
    '드라마/감동, 판타지/어드벤처', '코미디/유머, 판타지/어드벤처', 
    '액션/스릴러, 판타지/어드벤처', '드라마/감동, 액션/스릴러', 
    '액션/스릴러, 음악중심/주크박스'
]