# SKN03-FINAL-2Team
SKN03-FINAL-2Team

## Git 사용법
### Clone(처음 복사할 때)
```bash
git clone https://github.com/Leejoowon123/SKN_03_FINAL.git
git add .
``` 
### git pull(받아올 때)
```bash
git pull origin main
```
### git status(상태 확인: 올리거나 작업하기 전 항상 현 브랜치 & 상태 확인)
```bash
git status
git branch
```

### 새 브랜치 생성 및 push 방법
```bash
git checkout -b 브랜치명

# 변경사항 추가 및 커밋
git add .
git commit # i 클릭 후 커밋 메시지 작성 → 다 작성 후 esc 클릭 후 :wq 입력 후 엔터
git push origin 브랜치명 # origin 하면 자동으로 원격저장소에 같은 이름으로 push
```

## 문서 목록
- [모델 설명서](model/README.md)
- [데이터 명세서](model/READMEDATA.md)
- [데이터 파이프라인](model/DATAPIPELINE.md)

## 구현
### 추천 시스템 구현 화면
<p align="center">
  <img src="./READMEImages/추천_구현_화면_1.png" width="90%" />
  <img src="./READMEImages/추천_구현_화면_2.png" width="45%" height="300px" style="object-fit: contain; margin-bottom: 10px; margin-top: -10%;" />
  <img src="./READMEImages/추천_구현_화면_3.png" width="45%" height="300px" style="object-fit: contain; margin-bottom: 10px; margin-top: -10%;" />
</p>

### 추천 모델 성능 평가
<p align="center">
  <img src="./READMEImages/추천_모델_성능_평가.png" width="70%" />
</p>
<p align="center">
  <img src="./READMEImages/추천_모델_성능_평가_2.png" width="70%" />
</p>

