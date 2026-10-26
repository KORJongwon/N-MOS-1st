# NMOSFET 이상 진단 AX 재구현 및 확장 버전

이 저장소는 기존 Streamlit 기반의 단순 규칙 판정 프로그램을 바탕으로 **자기소개서에 서술한 전체 분석 흐름을 공부하고 시연할 수 있도록 사후에 확장한 코드**입니다. 기존 경진대회 당시 구현 기능을 입증하는 원본 증거가 아닙니다.

## 핵심 구성

1. `app.py`: Streamlit UI, MSA 게이트, lot 분석, 공정 이력 및 RAG 화면
2. `diagnostics.py`: 규격 판정, wafer/site 분석, Golden Wafer 반복 측정 Bias, crossed Gage R&R의 2-way random-effects ANOVA, P/T Ratio
3. `retrieval.py`: 로컬 문서에 대한 문자 TF-IDF 검색 후 선택적으로 Gemini 또는 OpenAI API에 검색 내용과 데이터 요약 전송
4. `knowledge/*.md`: 제공된 반도체 경진대회 발표자료 2페이지 및 4~13페이지 내용을 참고해 정리한 **검색용 요약문**
5. `sample_data/`: **모두 자체 생성한 시연용 가상 데이터**이며 실제 경진대회 데이터가 아님
6. `create_demo.py`: 가상 측정 CSV 생성

## 실행

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

[Streamlit Community Cloud](https://share.streamlit.io/)에서 GitHub 저장소를 선택하고 실행 파일을 `app.py`로 설정하면 배포할 수 있습니다. 데모 기능은 API 키 없이 동작합니다.

### 선택적 LLM 연결

실제 LLM으로 원인 후보를 생성하려면 Streamlit Cloud의 **Secrets**에 다음 둘 중 하나를 설정합니다.

```toml
OPENAI_API_KEY = "본인의_키"
# 또는
GEMINI_API_KEY = "본인의_키"
```

키는 **GitHub나 ZIP에 포함하지 않습니다.** API 제공사에 사용량 요금이 발생할 수 있습니다. API가 연결되지 않아도 로컬 문서 검색은 가능합니다. `retrieval.py`는 검색 자체가 LLM 추론이라고 주장하지 않으며, LLM 버튼을 눌렀을 때만 외부 모델에 실제 요청을 보냅니다.

## 업로드 데이터 형식

- Lot: `wafer_id,site,vth,idsat,idoff` (125행은 25장 × 5개 위치를 시연할 때 사용)
- Golden Wafer 반복 측정: `metric,reference,measured` (metric은 `vth`, `idsat`)
- 균형 교차 Gage R&R: `metric,part_id,operator,repeat,reading` (각 metric에서 2개 이상 part, operator, repeat. 실무에서는 보통 더 큰 표본을 사용)
- 공정 이력: 형식 자유. 예시에는 `case_id,process,parameter,target,observed,note`

`sample_data` 파일을 참조하면 상세 형식을 볼 수 있습니다. 실제 데이터는 관련 권한과 비밀 유지 의무를 확인한 뒤 사용하세요.

## 중요한 한계

- Golden Bias 및 Gage R&R 결과에 적용하는 10%, 30% 기준은 **시연용 의사결정 보조 기준**입니다. GRR/P/T는 6×GRR표준편차 / (상한−하한)이며, 실제 허용 여부는 제품과 측정 시스템의 요구 수준에 좌우됩니다.
- Idoff는 상한만 있는 단측 규격이므로 현재 코드에서 Idoff Gage R&R P/T를 계산하지 않습니다. Vth와 Idsat Gate가 양호해도 Idoff의 계측 신뢰성을 증명하지 못합니다.
- MSA 4-METHOD 전체, Golden Linearity/Stability, Gauge R&R의 완전한 측정 계획 적합성 판단 등은 구현하지 않았습니다.
- 위치별 평균과 범위 기반 경고는 초기 신호 탐지이며 공정 원인을 확정하지 않습니다.
- `knowledge/`는 경진대회 발표자료를 요약한 내부 참고 노트이며 권위 있는 외부 공정 이력의 증거가 아닙니다.
- LLM 답변은 환각 가능성이 있으므로 원인 후보만 제시하도록 안내했습니다. 실측 장비 로그, CD-SEM, SIMS 및 공정 이력으로 별도 검증해야 합니다.
- **사후 추가된 코드나 시연용 데이터가 과거에 실사용한 프로그램과 실제 수행한 일을 증명하지 않습니다.** 발표/면접 자료에는 구분해야 합니다.

## 테스트

```bash
python -m pytest -q
```
