"""파일 경로"""
file_path = "C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final"
per_raw = "per+raw.json" 
processed_data = "processed_data.json"
add_genre_file_name = "add_genre_story.json"
df_with_negatives_path = 'C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final/df_with_negatives.json'
picture_file_path = 'C:/SKN_3_MyProject/SKN_03_FINAL/READMEImages/Performance.jpg'
Score_Distribution_path = 'C:/SKN_3_MyProject/SKN_03_FINAL/READMEImages/Score_Distribution.jpg'
save_model_path = "C:/SKN_3_MyProject/SKN_03_FINAL/Data/Model/Recommend.h5"


# genre
unique_genres = [
    "대학로", "가족", "신화", "역사", "지역|창작"
]

# 삭제할 컬럼 목록
columns_to_drop = [
    "performance_id", "facility_id", "producer", "planner", 
    "host", "sponsor", "synopsis", "genre", "open_run", 
    "visit", "daehakro", "festival", "musical_create"
]


from langchain_core.agents import AgentFinish

def should_continue(data):
    """Checks if the agent outcome indicates the end of the conversation."""
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    else:
        return "continue"