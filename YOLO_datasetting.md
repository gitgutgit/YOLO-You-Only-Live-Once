
# yolo를 위한 데이터 세팅

## 1. 데이터 구조
txt 파일: 각 이미지의 라벨 데이터 (바운딩 박스 정보)
yaml 파일: 데이터셋 전체 설정 (경로, 클래스 정의 등)

구조 예시
game_dataset/
├── data.yaml          # 데이터셋 설정 파일
├── images/
│   ├── train/
│   │   ├── game_0001.jpg
│   │   ├── game_0002.jpg
│   │   └── game_0003.jpg
│   └── val/
│       ├── game_0100.jpg
│       └── game_0101.jpg
└── labels/
    ├── train/
    │   ├── game_0001.txt
    │   ├── game_0002.txt
    │   └── game_0003.txt
    └── val/
        ├── game_0100.txt
        └── game_0101.txt

## 2. 각 파일 내용 예시

### 2.1 txt 파일 예시; game__0001.txct
class_id x_center y_center width height
   ↓        ↓        ↓       ↓      ↓
   0      0.5234   0.3125  0.0850  0.1200


### 2.2yaml 파일 예시 (데이터셋 설정)

#### 2.2.1 간단 버전

##### 데이터셋 경로
path: /home/jeewon/game_dataset  # 절대 경로 또는 상대 경로
train: images/train
val: images/val

#####  클래스 정보
nc: 3  # number of classes
names: ['meteor', 'star', 'player']  # class names in order

#### 2.2.2 상세버전

path: ../datasets/game_dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3
names:
  0: meteor
  1: star
  2: player

# Optional: 추가 설정
download: false  # 자동 다운로드 비활성화


## 3. 그 외에 중요한 디테일

### 중요1 순서 일치시키기 


예시 1:
names: ['meteor', 'star', 'player'] 인 경우 0=metor ,1 =start 2= player에 맞게

### 중요2: 좌표는 정규화 가 되어있어야함 
예시1:
xcenter =xmain+(w/2) / imageWidth ...

