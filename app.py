"""NMOS 이상 진단 AX 재구현 및 확장 버전. 시연 데이터는 모두 가상 데이터."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import streamlit as st

from diagnostics import Specs, diagnose_lot, golden_accuracy, crossed_gage_rr, msa_gate
from retrieval import retrieve_documents, build_prompt, generate_with_openai, generate_with_gemini

BASE = Path(__file__).resolve().parent
DEMO = BASE / 'sample_data'

st.set_page_config(page_title='NMOS 신뢰성 선행 진단 | AX 재구현',page_icon='🔬',layout='wide')
st.title('🔬 NMOSFET 이상 진단 및 공정 원인 탐색')
st.caption('Python 판정 → Golden Wafer Accuracy → Gauge R&R 및 P/T → 125개 측정 데이터 → 위치별 분석 → 문서 검색(RAG) 및 선택적 LLM')
st.warning('이 앱은 경진대회 발표 내용을 바탕으로 나중에 보완한 **재구현 및 확장 버전**입니다. '
           '샘플 데이터와 공정 이력은 전부 시연용 가상 데이터이며, 경진대회 당시 실측 자료나 원본 개발 이력을 증명하지 않습니다.')


def read_csv(uploaded, fallback: Path | None):
    if uploaded is not None:
        return pd.read_csv(uploaded), False
    if fallback is not None:
        return pd.read_csv(fallback), True
    return None, False


def read_secret(name: str) -> str:
    env = os.getenv(name, '')
    if env:
        return env
    try:
        return str(st.secrets.get(name, '') or '')
    except (FileNotFoundError, KeyError, AttributeError):
        return ''


st.sidebar.header('데이터 및 판정 조건')
cases = {
    '사례 1: Vth 상승': 'CASE1_vth_high',
    '사례 2: Idsat 상승': 'CASE2_idsat_high',
    '사례 3: Idsat 산포 확대': 'CASE3_idsat_spread',
    '사례 4: Vth, Idsat 동시 감소': 'CASE4_both_low',
    '사례 5: Idoff 증가': 'CASE5_idoff_high',
}
case_label = st.sidebar.selectbox('시연 사례', list(cases))
case_id = cases[case_label]
use_sample = st.sidebar.checkbox('시연용 가상 데이터 사용',value=True)
lot_up = st.sidebar.file_uploader('Lot CSV (wafer_id, site, vth, idsat, idoff)',type='csv',key='lot')
golden_up = st.sidebar.file_uploader('Golden CSV (metric, reference, measured)',type='csv',key='golden')
grr_up = st.sidebar.file_uploader('Gauge R&R CSV (metric, part_id, operator, repeat, reading)',type='csv',key='grr')
history_up = st.sidebar.file_uploader('공정 이력 CSV (선택)',type='csv',key='history')

st.sidebar.caption('규격은 경진대회 예시에서 가져온 시작값이며, 실제 평가 규격과 일치하는지 확인이 필요합니다.')
vth_min = st.sidebar.number_input('Vth 하한 [V]',value=.25,format='%.4f',step=.01)
vth_max = st.sidebar.number_input('Vth 상한 [V]',value=.35,format='%.4f',step=.01)
idsat_min = st.sidebar.number_input('Idsat 하한 [uA/um]',value=850.,step=10.)
idsat_max = st.sidebar.number_input('Idsat 상한 [uA/um]',value=1150.,step=10.)
idoff_max = st.sidebar.number_input('Idoff 상한 [uA/um]',value=.5,min_value=.0001,format='%.4f',step=.1)
k_sigma = st.sidebar.select_slider('시연용 산포 범위 계수',options=[1.,2.,3.],value=1.,
                                    help='평균 ± k×표준편차와 규격을 비교하는 경험적 표시 방식이며 MSA 기준이나 불량률 추정과 다릅니다.')

if vth_min >= vth_max or idsat_min >= idsat_max:
    st.error('규격의 상한은 하한보다 커야 합니다.'); st.stop()
spec=Specs(vth_min,vth_max,idsat_min,idsat_max,idoff_max,k_sigma)
try:
    lot_df, lot_demo = read_csv(lot_up, (DEMO/f'{case_id}.csv') if use_sample else None)
    golden_df, golden_demo = read_csv(golden_up, DEMO/'golden_demo.csv' if use_sample else None)
    grr_df, grr_demo = read_csv(grr_up, DEMO/'grr_demo.csv' if use_sample else None)
    history_df, history_demo = read_csv(history_up, DEMO/'process_history_demo.csv' if use_sample else None)
    lot = diagnose_lot(lot_df,spec) if lot_df is not None else None
except (ValueError,KeyError,TypeError) as exc:
    st.error(f'데이터 오류: {exc}'); st.stop()

try:
    golden_result = golden_accuracy(golden_df,spec) if golden_df is not None else None
    grr_result = crossed_gage_rr(grr_df,spec) if grr_df is not None else None
    gate, gate_description = msa_gate(golden_result,grr_result)
except (ValueError,KeyError,TypeError) as exc:
    st.error(f'MSA 입력 오류: {exc}'); golden_result=None; grr_result=None
    gate,gate_description='미확인',str(exc)

if any([lot_demo,golden_demo,grr_demo,history_demo]):
    st.info('현재 선택된 자료에는 가상 예시가 포함됩니다. 실제 측정 및 장비 이력과 혼동하지 마세요.')

flow=st.tabs(['1. 신뢰성 게이트', '2. 소자 진단 및 위치 분석', '3. 공정 이력 + RAG', '4. 결과 저장 및 설명'])
with flow[0]:
    st.subheader('측정 신뢰성 선행 검증')
    if gate == '양호':
        st.success(f'{gate}: {gate_description}')
    elif gate in ('주의','미확인'):
        st.warning(f'{gate}: {gate_description}')
    else:
        st.error(f'{gate}: {gate_description}')
    a,b=st.columns(2)
    with a:
        st.markdown('#### Golden Wafer Bias / 반복 산포')
        if golden_result is None:
            st.info('자료 없음: Golden 측정 CSV를 올려주세요.')
        else:
            st.dataframe(golden_result,hide_index=True,use_container_width=True)
            st.caption('Bias = 측정 평균 - Golden 기준값. Accuracy 평가 참고 수치로 사용합니다.')
    with b:
        st.markdown('#### 교차 Gauge R&R 및 P/T')
        if grr_result is None:
            st.info('자료 없음: Gauge R&R CSV를 올려주세요.')
        else:
            st.dataframe(grr_result,hide_index=True,use_container_width=True)
            st.caption('균형 교차 ANOVA: P/T = 6 × GRR 표준편차 / 양측 규격 폭. 10%, 30% 경계는 참고용입니다.')
    st.caption('주의: Golden Bias 10% 기준도 시연용 설정입니다. Idoff는 단측 규격이며, 여기서 GRR/P/T를 측정하지 않습니다. '
               'MSA가 양호여도 공정 원인을 실험적으로 확정한 것은 아닙니다.')

with flow[1]:
    st.subheader('25장 웨이퍼, 각 5개 위치의 측정값 분석')
    if lot is None:
        st.info('Lot CSV 업로드가 필요합니다.');
    else:
        c1,c2,c3=st.columns(3)
        c1.metric('측정 행 수',f'{lot["n"]}개'); c2.metric('서로 다른 웨이퍼',f'{lot["wafer_count"]}장')
        c3.metric('측정 위치 종류',f'{lot["site_count"]}개')
        if lot['n']!=125 or lot['wafer_count']!=25 or lot['site_count']!=5:
            st.warning('입력 데이터가 자기소개서에서 언급한 25장 × 5위치 = 125개 구성과 다릅니다.')
        if lot['issues']:
            st.error('진단 플래그: '+', '.join(lot['issues']))
        else:
            st.success('현재 규격 및 산포 범위 조건에서 진단 플래그가 없습니다.')
        stat_rows=[]
        for metric,v in lot['stats'].items():
            stat_rows.append({'항목':metric.upper(),'평균':v['mean'],'표준편차':v['sd'],
                              '최솟값':v['min'],'최댓값':v['max'],'규격 이탈 측정 수':v['out_of_spec_count']})
        st.dataframe(pd.DataFrame(stat_rows),hide_index=True,use_container_width=True)
        st.caption('규격 이탈 개수는 개별 측정값 기준입니다. 평균 범위와 평균±kσ 기준도 함께 사용합니다.')
        st.markdown('#### 웨이퍼별 평균 추이')
        st.line_chart(lot['wafer_avg'].set_index('wafer_id')[['vth']])
        st.line_chart(lot['wafer_avg'].set_index('wafer_id')[['idsat']])
        st.markdown('#### 측정 위치별 평균')
        st.dataframe(lot['site_avg'],hide_index=True,use_container_width=True)
        with st.expander('입력 측정 원본 보기'):
            st.dataframe(lot_df,use_container_width=True)

with flow[2]:
    st.subheader('공정 이력 검증 후 원인 후보 탐색')
    if lot is None:
        st.info('먼저 측정 데이터를 입력해주세요.')
    elif gate != '양호':
        st.error('측정 신뢰성 게이트가 통과되지 않았습니다. 실제 변동으로 판정해 공정 원인을 추적하지 않습니다.')
        st.caption('MSA 자료를 보완하거나 다시 측정한 뒤 진행하세요. 검색/LLM 분석은 비활성화됩니다.')
    else:
        if history_df is not None:
            display_history=history_df.copy()
            if 'case_id' in display_history.columns and history_up is None:
                display_history=display_history[display_history.case_id==case_id]
            st.markdown('#### 확인 가능한 공정 이력')
            st.dataframe(display_history,hide_index=True,use_container_width=True)
            history_text=display_history.to_string(index=False)
        else:
            st.warning('실제 공정 이력을 업로드하지 않았습니다. 가능한 원인은 가설로만 표시합니다.')
            history_text=''; display_history=None
        tags=' '.join(lot['issues'])
        expanded_tags={'VTH 평균 높음':'Halo implant Delivered Dose Beam Current 문턱전압 상승',
          'IDSAT 평균 높음':'게이트 CD Idsat 상승', 'IDSAT 산포 범위 주의':'Idsat 산포 위치 별 Gate etch',
          'VTH 평균 낮음':'Vth 낮음 계면 결함 RTA 활성화', 'IDSAT 평균 낮음':'Idsat 낮음 RTA 계면 결함',
          'Idoff 규격 초과':'Idoff 높음 Halo DIBL'}
        query=tags+' '+' '.join(v for k,v in expanded_tags.items() if k in lot['issues'])
        retrieved=retrieve_documents(query or 'NMOSFET 공정 원인 측정 신뢰성')
        st.markdown('#### 1단계: 로컬 문서 검색 (RAG Retrieval)')
        for doc in retrieved:
            with st.expander(f'{doc["source"]} | 검색 유사도 {doc["score"]:.3f}'):
                st.text(doc['text'])
        st.caption('검색은 로컬 기술 문서의 문자 단위 TF-IDF 검색입니다. 문서가 검색되었다는 사실만으로 원인이 검증된 것은 아닙니다.')
        st.markdown('#### 2단계: 실제 생성형 AI를 선택적으로 호출')
        provider=st.radio('분석 모드', ['문서 검색만 (API 호출 안 함)','OpenAI API','Gemini API'],horizontal=True)
        keyname='OPENAI_API_KEY' if provider=='OpenAI API' else 'GEMINI_API_KEY'
        key=read_secret(keyname) if provider!='문서 검색만 (API 호출 안 함)' else ''
        if provider!='문서 검색만 (API 호출 안 함)' and not key:
            st.info(f'{keyname}가 등록되지 않았습니다. Streamlit Secrets를 설정해야 실제 LLM 추론을 실행할 수 있습니다.')
        if st.button('원인 후보 및 추가 확인 항목 생성',disabled=provider=='문서 검색만 (API 호출 안 함)' or not key):
            prompt=build_prompt(lot,history_text,retrieved,lot_demo or history_demo)
            try:
                with st.spinner('검색 자료와 입력 데이터를 근거로 가설을 생성 중'):
                    output=(generate_with_openai(prompt,key) if provider=='OpenAI API'
                            else generate_with_gemini(prompt,key))
                st.session_state['latest_llm']={'case':case_id,'text':output,'provider':provider}
            except Exception as exc:
                st.error(f'LLM 요청 실패: {exc}')
        report=st.session_state.get('latest_llm')
        if report is not None and report['case']==case_id:
            st.markdown(f'**{report["provider"]} 생성 결과 (검증 전 가설)**')
            st.markdown(report['text'])
        if provider=='문서 검색만 (API 호출 안 함)':
            st.info('현재는 검색 결과만 표시하며 생성형 AI가 원인 후보를 새로 추론하지 않습니다.')
        st.warning('Delivered Dose, Beam Current 등이 표에 표시되어도 실제 계측 장비 로그와 대조하기 전에는 공정 원인 확정이 아닙니다.')

with flow[3]:
    st.subheader('재현 가능한 진단 결과 저장')
    text=[f'[NMOSFET 진단 보고서: {datetime.now():%Y-%m-%d %H:%M}]',
          '버전: 경진대회 자료를 바탕으로 사후 재구현 및 확장한 프로그램',
          f'데이터 성격: {"모의/시연용 포함" if any([lot_demo,golden_demo,grr_demo,history_demo]) else "사용자 제공 (진위 확인하지 않음)"}',
          f'계측 신뢰성 게이트: {gate} - {gate_description}']
    if golden_result is not None:
        text.extend(['\n[Golden Bias]',golden_result.to_string(index=False)])
    if grr_result is not None:
        text.extend(['\n[교차 Gauge R&R / P/T]',grr_result.to_string(index=False)])
    if lot is not None:
        text.extend(['\n[Lot]',f"측정 수: {lot['n']}, 웨이퍼: {lot['wafer_count']}, 위치: {lot['site_count']}",
                     f"규격 플래그: {', '.join(lot['issues']) or '없음'}"])
        for metric,v in lot['stats'].items():
            text.append(f"{metric}: mean={v['mean']:.5g}, std={v['sd']:.5g}, 규격 이탈={v['out_of_spec_count']}개")
    text.extend(['\n[한계]','규격 판정은 Python 규칙 기반. MSA는 사용자가 올린 자료만으로 계산.',
                 'Idoff의 단측 MSA, MSA Linearity 및 Stability 등은 구현하지 않음.',
                 '공정 원인 확정은 장비 로그 및 별도 실측으로 확인해야 함.'])
    latest=st.session_state.get('latest_llm')
    if latest is not None and latest['case']==case_id:
        text.extend(['\n[참고: 생성형 AI의 미검증 원인 가설]',latest['text']])
    st.download_button('진단 보고서 TXT 다운로드',data='\n'.join(text),
                       file_name=f'nmos_ax_{case_id.lower()}_report.txt',mime='text/plain')
    st.markdown('#### 실제 제출 자료에서 구분해야 할 내용')
    st.write('**기존 구현 코드에서 확인된 기능:** 규격 판정, 평균·산포 기반 플래그, 정적 원인 안내, Streamlit 화면 및 TXT 저장.')
    st.write('**이번 확장에 새로 구현한 기능:** 125개 CSV 데이터, Golden Bias, 교차 Gage R&R 및 P/T, 웨이퍼와 위치별 집계, 로컬 RAG 문서 검색, 선택적 실제 LLM API 호출 및 공정 이력 업로드.')
    st.write('**여전히 실증되지 않은 사항:** 경진대회 당시 RAG/LLM을 실제 구동했는지, 실제 125개 원자료 및 장비 로그와의 일치, 공정 원인 확정.')
