"""파일 경로"""
file_path = "C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final"
model_file_path = "C:/SKN_3_MyProject/SKN_03_FINAL/Data/Model"
picture_file_path = "C:/SKN_3_MyProject/SKN_03_FINAL/READMEImages"
input_file_name = "merged_output.json"
output_file_name = "processed_performance_details.json"
add_genre_file_name = "add_genre_story.json"
last_processing_file_name = "final_processing.json"
model_name = "DeepFM.pkl"
embedding_file = "C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final/embedding.json"
embedding_file_name = "embedding.json"
mapping_file_name = "mapping.json"


"""장르"""
genre_priority= {
            '드라마/감동': 1,
            '코미디/유머': 2,
            '액션/스릴러': 3,
            '판타지/어드벤처': 4,
            '음악중심/주크박스': 5,
}

"""요일"""
day = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

"""제거할 컬럼"""
columns_to_remove = [
    # 'genre'
    'synopsis', 'open_run', 'visit', 'child', 'daehakro',
    'festival', 'musical_create', 'sponsor', 'planner',
    'performance_id', 'facility_id','place_y', 'end_date_y',
    'start_date_y', 
]

"""NULL 처리 컬럼"""
columns_with_empty_values = ['cast', 'producer', 'host', 'editor', 'runtime','prfdtcnt','seatcnt', 'tickets', 'percentage']

# columns_to_remove = [
#     'place_y', 'end_date_y', 'start_date_y'
# ]
# columns_with_empty_values = ['prfdtcnt','seatcnt', 'tickets', 'percentage']

"""모델용"""
# deep_features = [
#             "place_x", "runtime", "musical_license", "period", "day_time_1", "day_time_2", "day_time_3",
#             "day_time_4", "day_time_5", "day_time_6", "day_time_7","editor_1", "editor_2", "editor_3", 
#             "ticket_price", "tickets", "seatcnt", "prfdtcnt"
# ]
# fm_features = [
#             "genre", "cast_1", "cast_2", "cast_3", "cast_4", "cast_5", "cast_6", "cast_7", "cast_8" 
# ]
# # target_column = "percentage"
target_column = "target"
fm_features = ["cast_id", "genre", "title"]
deep_features = ["editor_combined_id", "percentage", "musical_license", "period", "ticket_price", "day_vector", "time_category"]
expected_keys = ["cast_id", "genre", "title", 
                    "editor_combined_id", "percentage", 
                    "musical_license", "period", "ticket_price", 
                    "day_vector", "time_category","actor_genre_preference",
                    "actor_sales_influence","day_time_interaction"
                    ]
int_columns = ['genre', 'musical_license', 'time_category', 'target']
float_columns = ['percentage', 'period', 'ticket_price', 'day_vector',
                 "time_category","actor_genre_preference",
                "actor_sales_influence","day_time_interaction"]