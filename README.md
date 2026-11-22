# WM-811K 웨이퍼 결함 분류 모델 비교와 품질 분석

경희대학교 소프트웨어랩(EE370) Term Project입니다. WM-811K 웨이퍼 맵을 정상 패턴을 포함한 9개 유형으로 분류하고, 결함의 위치·밀도·형상을 수치화한 모델 4종과 이미지 기반 CNN 2종의 성능을 비교했습니다. 소프트웨어랩 7강 ML의 분류와 8강 CNN의 합성곱·풀링을 적용했습니다.

- 저장소: [WooSeungJi/WM811K-Project](https://github.com/WooSeungJi/WM811K-Project)
- 재현용 노트북: [Ram_절약_WM_811K_Wafer_Map.ipynb](Ram_%EC%A0%88%EC%95%BD_WM_811K_Wafer_Map.ipynb)
- 초기 실험: [머신러닝_WM_811K_Wafer_Map.ipynb](%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_WM_811K_Wafer_Map.ipynb)

## 데이터와 실험 설계

[WM-811K 데이터셋](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)의 `LSWMD.pkl`을 사용했습니다. 원본 웨이퍼 맵에서 값 0은 웨이퍼 밖, 1은 정상 다이, 2는 불량 다이를 뜻합니다. 라벨이 없는 샘플을 제외하고 정상 클래스(`none`)를 15,000개로 다운샘플링했습니다. 난수 시드 42로 층화 분할한 뒤 학습 32,415장, 검증 4,052장, 테스트 4,052장을 사용했습니다.

- **특징 기반 분류** — 원본 맵에서 전체 결함률, 중심부·외곽부 결함률, 결함의 위치·형상 등을 추출해 Logistic Regression, Decision Tree, Random Forest, Linear SVM을 비교했습니다.
- **이미지 기반 분류** — Light CNN은 최근접 보간으로 변환한 64×64 단일 채널 맵을 입력받습니다. ResNet50은 3채널 입력으로 변환하고 합성곱 기반을 고정한 상태에서 분류기를 학습했습니다. 학습 중에는 웨이퍼 맵을 배치 단위로 읽어 메모리 사용량을 줄였습니다.

## Colab에서 실행하기

실험 환경은 **Google Colab(Linux, Python 3)**입니다. TensorFlow/Keras, NumPy, pandas, scikit-learn, SciPy, joblib, Matplotlib을 사용합니다. GPU는 CNN을 다시 학습할 때 사용할 수 있습니다.

1. 데이터셋에서 `LSWMD.pkl`을 다운로드해 본인 Google Drive의 `내 드라이브/ML Project/WM811K/`에 넣습니다.
2. [Colab에서 재현용 노트북 열기](https://colab.research.google.com/github/WooSeungJi/WM811K-Project/blob/main/Ram_%EC%A0%88%EC%95%BD_WM_811K_Wafer_Map.ipynb)를 누릅니다. 첫 코드 셀에서 Google Drive 연결을 승인합니다. 다른 폴더를 사용할 때는 같은 셀의 `DATA_PATH`와 `OUTPUT_DIR`을 수정합니다. 기본값은 다음과 같습니다.

   ```python
   DATA_PATH = "/content/drive/MyDrive/ML Project/WM811K/LSWMD.pkl"
   OUTPUT_DIR = "/content/drive/MyDrive/ML Project/WM811K/outputs"
   ```

3. **데이터 전처리와 분할 → Light CNN 학습 및 평가 → Feature Engineering과 기준 모델 비교 → ResNet50 학습 및 평가 → 최종 비교와 Edge-Ring 사례** 순서로 셀을 실행합니다. 기존 실행 결과는 GitHub의 노트북에도 남아 있습니다.
4. 실행 후 `outputs/`에서 `light_cnn_result.csv`와 `final_light_cnn.keras`를 확인합니다. `outputs/resnet50_light_experiment/`에는 ResNet50 결과가, `outputs/report_figures/`에는 Edge-Ring 사례의 그림과 수치가 저장됩니다.

Colab 런타임에 설치된 버전은 아래 셀로 확인할 수 있습니다. 원본 노트북에는 전체 패키지 버전이 기록되어 있지 않습니다.

```python
import platform
from importlib.metadata import version

print("OS:", platform.platform())
print("Python:", platform.python_version())
for package in ("tensorflow", "keras", "numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib"):
    print(package, version(package))
```

## 테스트 결과

아래 수치는 기존 Colab 실험의 테스트 결과입니다. 클래스별 표본 수가 다르므로 전체 정확도와 함께 각 클래스에 같은 비중을 주는 Macro F1을 비교했습니다.

| 모델 | Accuracy | Balanced Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.8460 | 0.8175 | 0.7613 |
| Decision Tree | 0.8292 | 0.7496 | 0.7505 |
| Random Forest | 0.8959 | 0.8046 | 0.8224 |
| Linear SVM | 0.8549 | 0.7777 | 0.7386 |
| **Light CNN** | **0.9099** | **0.8845** | **0.8663** |
| ResNet50 | 0.7549 | 0.7707 | 0.6867 |

Light CNN의 Macro F1이 0.8663으로 가장 높았고, 특징 기반 모델에서는 Random Forest가 0.8224로 가장 높았습니다. 혼동행렬에서는 Loc과 Edge-Loc처럼 결함이 국소적으로 모이는 유형을 구분하기 어려운 사례가 나타났습니다.

Edge-Ring 사례(테스트 샘플 31806)에서 Light CNN과 Random Forest는 모두 Edge-Ring을 예측했습니다. 영역별 결함률은 중심부 3.04%, 외곽부 20.19%로, 외곽에 결함이 집중된 모습을 원본 맵과 결함 마스크에서 함께 확인했습니다. 이 사례에서는 외곽부 공정 균일도와 웨이퍼 이송 조건을 추가 점검 항목으로 제안했습니다.

## AI 활용 및 검증

ChatGPT(Codex)를 계획서와 README 초안 정리, 노트북 코드 확인, 저장된 Light CNN의 샘플 추론 검증에 사용했습니다. README의 경로·설정·주요 지표는 노트북 코드와 기존 실행 출력에서 확인했고, 저장된 모델에 별도의 WM-811K 샘플 한 장을 넣어 상위 3개 예측이 나오는지 확인했습니다.

## 참고 자료

- 소프트웨어랩 7강 ML, 8강 CNN 강의자료
- [WM-811K Wafer Map 데이터셋](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- Wu, Jang, Chen, [Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets](https://doi.org/10.1109/TSM.2014.2364237), IEEE Transactions on Semiconductor Manufacturing, 2015.
