import pandas as pd
import ast

# 1) 원본 CSV 읽기
df = pd.read_csv('data/ptbxl/ptbxl_database.csv')

# 2) 모든 scp_code 키 집합 수집
all_codes = set()
for entry in df['scp_codes']:
    # entry 예시: "{426783006: 1, 713427006: 1, …}"
    code_dict = ast.literal_eval(entry)
    all_codes |= set(code_dict.keys())
all_codes = sorted(all_codes)
print(f"총 {len(all_codes)}개의 진단 코드 발견")

# 3) 코드 → 인덱스 매핑
code2idx = {code: idx for idx, code in enumerate(all_codes)}

# 4) 각 레코드에 대해 multi-label 벡터 생성
def make_label_vector(entry):
    code_dict = ast.literal_eval(entry)
    vec = [0] * len(all_codes)
    for code in code_dict.keys():
        vec[code2idx[code]] = 1
    return vec

df['labels'] = df['scp_codes'].apply(make_label_vector)

# 5) 새로운 메타 CSV로 저장
df.to_csv('data/ptbxl_meta_with_labels.csv', index=False)
print("labels 컬럼이 추가된 메타데이터를 data/ptbxl_meta_with_labels.csv 에 저장했습니다.")