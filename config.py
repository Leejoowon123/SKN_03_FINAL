"""파일 경로"""
file_path = "C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final"
input_file_name = "performance_details.json"
output_file_name = "processed_performance_details.json"
add_genre_file_name = "add_genre_story.json"
last_processing_file_name = "final_processing.json"

"""제거할 컬럼"""
columns_to_remove = [
    'genre', 'synopsis', 'open_run', 'visit', 'child', 'daehakro',
    'festival', 'musical_create', 'sponsor', 'planner',
    'performance_id', 'facility_id'
]
"""NULL 처리 컬럼"""
columns_with_empty_values = ['cast', 'producer', 'host', 'editor', 'runtime']