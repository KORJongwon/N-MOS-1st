"""시연 전용 모의 데이터 생성기. 실제 대회 데이터가 아닙니다."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/'sample_data'


def make_demo():
    OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261026)
    scenarios={
      'CASE1_vth_high': {'vth':.385,'idsat':1005,'idoff':.25,'vs':.010,'is':30},
      'CASE2_idsat_high':{'vth':.30,'idsat':1300,'idoff':.24,'vs':.010,'is':35},
      'CASE3_idsat_spread':{'vth':.30,'idsat':1000,'idoff':.24,'vs':.010,'is':180},
      'CASE4_both_low':{'vth':.215,'idsat':800,'idoff':.27,'vs':.008,'is':25},
      'CASE5_idoff_high':{'vth':.29,'idsat':1000,'idoff':1.45,'vs':.008,'is':30},
    }
    for case_name, v in scenarios.items():
        rows=[]
        for wafer in range(1,26):
            wafer_shift=rng.normal()
            for site in ['C','N','S','E','W']:
                # 시연용으로 일부 위치 편차를 주어 wafer/site 분석을 확인할 수 있도록 함.
                spatial= 1 if site in ('N','E') else (-1 if site in ('S','W') else 0)
                rows.append({'wafer_id':f'W{wafer:02d}','site':site,
                        'vth':v['vth']+v['vs']*(0.35*wafer_shift+0.2*spatial+rng.normal(0,.55)),
                        'idsat':v['idsat']+v['is']*(0.35*wafer_shift+.25*spatial+rng.normal(0,.65)),
                        'idoff':max(0,v['idoff']+(.05 if v['idoff']<1 else .20)*rng.normal())})
        pd.DataFrame(rows).to_csv(OUT/f'{case_name}.csv',index=False,encoding='utf-8-sig')
    golden=[]; gage=[]
    for metric,reference in [('vth',.300),('idsat',1000.0)]:
        golden_noise=.0015 if metric=='vth' else 3.0
        for k in range(10):
            golden.append({'metric':metric,'reference':reference,'measured':reference+rng.normal(0,golden_noise)})
        part_scale=.012 if metric=='vth' else 28
        repeat_scale=.001 if metric=='vth' else 2.4
        operator_scale=.0006 if metric=='vth' else 1.0
        for part in range(1,11):
            part_val=reference+rng.normal(0,part_scale)
            for i,operator in enumerate(['A','B','C']):
                for rep in [1,2]:
                    gage.append({'metric':metric,'part_id':f'P{part:02d}','operator':operator,
                                 'repeat':rep,'reading':part_val+(i-1)*operator_scale+rng.normal(0,repeat_scale)})
    pd.DataFrame(golden).to_csv(OUT/'golden_demo.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(gage).to_csv(OUT/'grr_demo.csv',index=False,encoding='utf-8-sig')
    history = [
       {'case_id':'CASE1_vth_high','process':'Halo implant','parameter':'Delivered Dose (normalized)','target':1.0,'observed':1.08,'note':'시연용 입력, 실제 이력 아님'},
       {'case_id':'CASE1_vth_high','process':'Ion implanter','parameter':'Beam Current (normalized)','target':1.0,'observed':1.05,'note':'시연용 입력, 실제 이력 아님'},
       {'case_id':'CASE2_idsat_high','process':'Gate lithography','parameter':'Gate CD (normalized)','target':1.0,'observed':.95,'note':'시연용 입력, 실제 이력 아님'},
       {'case_id':'CASE3_idsat_spread','process':'Gate etch','parameter':'CD spatial nonuniformity (normalized)','target':1.0,'observed':1.3,'note':'시연용 입력, 실제 이력 아님'},
       {'case_id':'CASE4_both_low','process':'Anneal','parameter':'Peak Temperature (normalized)','target':1.0,'observed':.98,'note':'시연용 입력, 실제 이력 아님'},
       {'case_id':'CASE5_idoff_high','process':'Halo implant','parameter':'Tilt angle (normalized)','target':1.0,'observed':1.04,'note':'시연용 입력, 실제 이력 아님'},
    ]
    pd.DataFrame(history).to_csv(OUT/'process_history_demo.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    make_demo()
