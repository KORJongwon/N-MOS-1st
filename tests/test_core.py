from pathlib import Path
import pandas as pd
import pytest
from diagnostics import Specs, golden_accuracy, crossed_gage_rr, msa_gate, diagnose_lot, validate_lot
from retrieval import retrieve_documents, build_prompt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'sample_data'

def test_all_five_cases():
    keys=('VTH 평균 높음','IDSAT 평균 높음','IDSAT 산포 범위 주의','VTH 평균 낮음','Idoff 규격 초과')
    files=sorted(DATA.glob('CASE*.csv'))
    assert len(files)==5
    for file,expected in zip(files,keys):
        result=diagnose_lot(pd.read_csv(file),Specs())
        assert expected in result['issues'], (file.name,result['issues'])
        assert result['n']==125 and result['wafer_count']==25 and result['site_count']==5

def test_msa_good():
    golden=golden_accuracy(pd.read_csv(DATA/'golden_demo.csv'),Specs())
    grr=crossed_gage_rr(pd.read_csv(DATA/'grr_demo.csv'),Specs())
    assert set(grr.metric)=={'vth','idsat'}
    assert (grr['P/T Ratio']<.10).all()
    assert msa_gate(golden,grr)[0]=='양호'

def test_gate_blocks_missing_and_bad_grr():
    golden=golden_accuracy(pd.read_csv(DATA/'golden_demo.csv'),Specs())
    grr=crossed_gage_rr(pd.read_csv(DATA/'grr_demo.csv'),Specs())
    assert msa_gate(golden,None)[0]=='미확인'
    grr.loc[grr.metric=='vth','P/T Ratio']=.51
    assert msa_gate(golden,grr)[0]=='부적합'

def test_retrieval_and_prompt():
    lot=diagnose_lot(pd.read_csv(DATA/'CASE1_vth_high.csv'),Specs())
    docs=retrieve_documents('Vth 상승 Halo implant Delivered Dose Beam Current')
    assert docs and len(docs)<=4
    prompt=build_prompt(lot,'Delivered Dose 1.08 (모의 값)',docs,True)
    assert '생성한 시연용 가상 데이터' in prompt

def test_invalid_lot():
    frame=pd.read_csv(DATA/'CASE1_vth_high.csv')
    frame.loc[0,'vth']='bad'
    with pytest.raises(ValueError): validate_lot(frame)

def test_bad_grr_missing_cell():
    grr=pd.read_csv(DATA/'grr_demo.csv').iloc[:-1]
    with pytest.raises(ValueError): crossed_gage_rr(grr,Specs())
