# Graph-Structure-Aware Shortest-Path Algorithm Benchmark & Selection Rule

본 저장소는 다양한 그래프 구조에서 최단경로 알고리즘의 실행 성능을 체계적으로 비교하고,  
그래프 구조적 특성(feature)을 기반으로 **가장 빠른 알고리즘을 자동 선택**하는 방법을 재현하기 위한 연구 코드입니다.

본 코드는 논문 *Algorithms (7).pdf*의 실험 및 결과를 **완전히 재현 가능**하도록 설계되었습니다.

---

## 1. Overview

### Algorithms
- DIJKSTRA
- A*
- BI_DIJKSTRA (Bidirectional Dijkstra)
- ALT (A* + Landmarks + Triangle inequality)

### Graph Families
- ER (Erdős–Rényi)
- BA (Barabási–Albert)
- WS (Watts–Strogatz)
- GRID
- COMM (Community-based)

### Main Goals
1. 그래프 패밀리별 알고리즘 성능 비교
2. 그래프 구조 feature와 알고리즘 성능의 관계 분석
3. **그래프 인스턴스 단위 자동 알고리즘 선택**
4. 해석 가능한 선택 규칙(Decision Tree) 도출

---

## 2. Requirements

- Python ≥ 3.10
- OS: macOS / Linux / Windows
- 주요 라이브러리
  - numpy
  - pandas
  - scikit-learn
  - matplotlib

### Install

```bash
conda create name python=3.10
conda activate name
pip install -r requirements.txt
```
---

## 3. Repository Structure

```
.
├── run_experiments.py              # 전체 실험 파이프라인 실행
├── shortest_path_algs.py           # 최단경로 알고리즘 구현
├── graph_families.py               # 그래프 생성기
├── graph_features.py               # 그래프 구조 feature 계산
├── eval_selection_groupkfold.py    # ML 기반 알고리즘 선택 평가
├── derive_rule.py                  # 결정트리 기반 선택 규칙 추출
├── make_tables.py                  # 논문 표 생성
├── make_plots.py                   # 논문 그림 생성
├── plot_feature_importance.py      # feature 중요도 시각화
├── stats_tests.py                  # 통계 검정
├── Algorithms (7).pdf              # 논문 본문
├── out_paper/                      # 실험 결과 CSV
├── tables_paper/                   # 표 출력
├── plots_paper/                    # 그림 출력
└── README.md
```

---

## 4. Run Experiments (Raw CSV 생성)

### 4.1 Basic Mode (A* vs DIJKSTRA)

```bash
python run_experiments.py \
  --mode basic \
  --out out_paper/results_basic.csv
```

### 4.2 Extended Mode (A*, ALT, BI_DIJKSTRA, DIJKSTRA)

```bash
python run_experiments.py \
  --mode extended \
  --out out_paper/results_extended.csv
```

### Output

실행 결과는 자동으로 timestamp가 붙은 CSV로 저장됩니다.

예:
- results_basic_20260208_085258.csv
- results_extended_20260208_085647.csv

---

## 5. CSV Format (중요)

CSV에는 최소한 다음 컬럼들이 포함되어야 합니다.

### Graph Instance Keys
- family
- n
- p
- weight
- seed
- graph_idx

### Runtime
- algo
- cpu_sec

### Graph Features (가능한 경우)
- avg_degree
- degree_cv
- clustering
- diam_est
- aspl_est
- w_mean
- w_std
- density
- n_edges

---

## 6. Table & Plot Generation

### 6.1 Tables

```bash
python make_tables.py \
  --results out_paper/results_basic.csv \
  --mode basic \
  --outdir tables_paper
```

```bash
python make_tables.py \
  --results out_paper/results_extended.csv \
  --mode extended \
  --outdir tables_paper2
```

### 6.2 Plots

```bash
python make_plots.py \
  --results out_paper/results_basic.csv \
  --mode basic \
  --outdir plots_paper
```

```bash
python make_plots.py \
  --results out_paper/results_extended.csv \
  --mode extended \
  --outdir plots_paper2
```

---

## 7. Statistical Tests

알고리즘 간 성능 차이의 통계적 유의성을 검정합니다.

```bash
python stats_tests.py \
  --results out_paper/results_basic.csv \
  --mode basic \
  --out tables_paper/stats_basic.txt
```

```bash
python stats_tests.py \
  --results out_paper/results_extended.csv \
  --mode extended \
  --out tables_paper2/stats_extended.txt
```

---

## 8. Algorithm Selection Evaluation (GroupKFold)

그래프 인스턴스 단위 데이터 누수를 방지하기 위해 GroupKFold를 사용합니다.

### 8.1 Basic (Binary Classification)

A*가 DIJKSTRA보다 빠른지 예측

```bash
python eval_selection_groupkfold.py \
  --results out_paper/results_basic.csv \
  --mode basic
```

출력:
- Accuracy
- F1-score
- ROC-AUC
- Baseline
- Confusion Matrix

---

### 8.2 Extended (Multiclass Classification)

가장 빠른 알고리즘 선택

```bash
python eval_selection_groupkfold.py \
  --results out_paper/results_extended.csv \
  --mode extended
```

출력:
- Accuracy
- Macro F1
- Weighted F1
- Baseline
- Classification Report

---

## 9. Derive Interpretable Selection Rule

얕은 결정트리로 사람이 읽을 수 있는 선택 규칙을 생성합니다.

### 9.1 Basic Rule

```bash
python derive_rule.py \
  --results out_paper/results_basic.csv \
  --outtxt tables_paper/derived_rule_basic.txt \
  --mode basic \
  --tree_depth 3
```

### 9.2 Extended Rule

```bash
python derive_rule.py \
  --results out_paper/results_extended.csv \
  --outtxt tables_paper2/derived_rule_extended.txt \
  --mode extended \
  --tree_depth 3
```

---

## 10. Reproducibility Notes

- 모든 실험은 중앙값(median) 기준으로 알고리즘 성능을 비교합니다.
- 선택 모델 평가는 query 단위가 아닌 graph instance 단위로 수행합니다.
- 동일한 그래프 인스턴스가 train/test에 동시에 포함되지 않도록 GroupKFold를 사용합니다.
- 랜덤 시드는 고정되어 있습니다.

---

## 11. Quick Reproduction (최소 재현)

```bash
python run_experiments.py --mode basic --out out_paper/results_basic.csv
python eval_selection_groupkfold.py --results out_paper/results_basic.csv --mode basic
python derive_rule.py --results out_paper/results_basic.csv --outtxt tables_paper/derived_rule_basic.txt --mode basic --tree_depth 3
```

---

## 12. Common Issues

### Feature가 NaN으로 제거되어 학습 데이터가 0개인 경우
- graph_features.py가 정상 실행되는지 확인
- derive_rule.py에서 사용하는 feature 목록을 줄이거나 조정

### A* 컬럼명이 A로 저장된 경우
- 코드 내부에서 자동으로 처리됨

---

## 13. Citation

본 코드 또는 결과를 사용할 경우 논문을 인용해 주십시오.

Graph-Structure-Aware Selection of Shortest Path Algorithms,  
Algorithms (2026)
