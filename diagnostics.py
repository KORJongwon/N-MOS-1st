"""측정 데이터 규격 판정과 교차 Gage R&R (balanced random-effects ANOVA)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

METRICS = ('vth', 'idsat', 'idoff')

@dataclass(frozen=True)
class Specs:
    vth_min: float = 0.25
    vth_max: float = 0.35
    idsat_min: float = 850.0
    idsat_max: float = 1150.0
    idoff_max: float = 0.5
    sigma_k: float = 1.0

    def tolerance(self, metric: str) -> Optional[float]:
        if metric == 'vth':
            return self.vth_max - self.vth_min
        if metric == 'idsat':
            return self.idsat_max - self.idsat_min
        return None  # 단측 규격인 Idoff에는 P/T를 임의 적용하지 않는다.


def validate_lot(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'wafer_id', 'site', 'vth', 'idsat', 'idoff'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Lot CSV 필수 열 누락: {', '.join(sorted(missing))}")
    df = frame.copy()
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[list(METRICS)].isna().any().any():
        raise ValueError('측정값에 빈 값 또는 숫자가 아닌 값이 있습니다.')
    if not np.isfinite(df[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError('무한대 값은 사용할 수 없습니다.')
    if (df['idoff'] < 0).any():
        raise ValueError('Idoff는 음수일 수 없습니다.')
    if df[['wafer_id', 'site']].isna().any().any() or df.duplicated(['wafer_id', 'site']).any():
        raise ValueError('wafer_id, site는 비어 있으면 안 되며 동일 조합이 중복될 수 없습니다.')
    if len(df) < 5:
        raise ValueError('측정 데이터가 너무 적습니다. 최소 5행이 필요합니다.')
    return df


def diagnose_lot(df: pd.DataFrame, s: Specs) -> dict:
    df = validate_lot(df)
    limits = {'vth': (s.vth_min, s.vth_max), 'idsat': (s.idsat_min, s.idsat_max)}
    stats = {}
    issues = []
    for key in METRICS:
        arr = df[key].to_numpy(dtype=float)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        stats[key] = {'mean': mean, 'sd': sd, 'min': float(arr.min()), 'max': float(arr.max())}
        if key == 'idoff':
            bad_count = int(np.sum(arr > s.idoff_max))
            stats[key]['out_of_spec_count'] = bad_count
            if bad_count:
                issues.append('Idoff 규격 초과')
            continue
        lo, hi = limits[key]
        bad_count = int(np.sum((arr < lo) | (arr > hi)))
        stats[key]['out_of_spec_count'] = bad_count
        if mean < lo:
            issues.append(f'{key.upper()} 평균 낮음')
        elif mean > hi:
            issues.append(f'{key.upper()} 평균 높음')
        if lo <= mean <= hi and (mean - s.sigma_k * sd < lo or mean + s.sigma_k * sd > hi):
            issues.append(f'{key.upper()} 산포 범위 주의')
    wafer_avg = df.groupby('wafer_id', as_index=False)[list(METRICS)].mean()
    site_avg = df.groupby('site', as_index=False)[list(METRICS)].mean()
    return {'issues': list(dict.fromkeys(issues)), 'stats': stats, 'wafer_avg': wafer_avg,
            'site_avg': site_avg, 'n': len(df), 'wafer_count': df.wafer_id.nunique(),
            'site_count': df.site.nunique()}


def golden_accuracy(frame: pd.DataFrame, spec: Specs) -> pd.DataFrame:
    required = {'metric', 'reference', 'measured'}
    if not required.issubset(frame.columns):
        raise ValueError('Golden CSV 필수 열: metric, reference, measured')
    df = frame.copy()
    if not df['metric'].isin(('vth', 'idsat')).all():
        raise ValueError('Golden 분석은 현재 vth, idsat 두 측정 항목을 지원합니다.')
    for col in ('reference', 'measured'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['reference', 'measured']].isna().any().any():
        raise ValueError('Golden 값에 비정상적인 값이 있습니다.')
    records = []
    for metric, group in df.groupby('metric'):
        if len(group) < 3:
            raise ValueError(f'{metric}: Golden 반복 측정은 최소 3회가 필요합니다.')
        if group.reference.nunique() != 1:
            raise ValueError(f'{metric}: Golden 기준값은 동일해야 합니다.')
        reference = float(group.reference.iloc[0]); samples = group.measured.to_numpy(dtype=float)
        bias = float(samples.mean()-reference)
        sd = float(samples.std(ddof=1))
        tol = spec.tolerance(metric)
        records.append({'metric': metric, '반복 수': len(group), '기준값': reference,
                        '측정 평균': float(samples.mean()), 'Bias': bias,
                        '반복 표준편차': sd, '|Bias|/허용폭 (%)': abs(bias)/tol*100})
    return pd.DataFrame(records)


def crossed_gage_rr(frame: pd.DataFrame, spec: Specs) -> pd.DataFrame:
    """균형 교차 설계 2-way random effects ANOVA: operator, part, interaction, repeat."""
    columns = {'metric', 'part_id', 'operator', 'repeat', 'reading'}
    if not columns.issubset(frame.columns):
        raise ValueError('GRR CSV 필수 열: metric, part_id, operator, repeat, reading')
    df = frame.copy()
    if not df.metric.isin(('vth','idsat')).all():
        raise ValueError('GRR 분석은 vth와 idsat만 지원합니다 (양측 규격).')
    df['reading'] = pd.to_numeric(df['reading'], errors='coerce')
    if df.reading.isna().any() or df.duplicated(['metric','part_id','operator','repeat']).any():
        raise ValueError('GRR 중복 측정 열 또는 숫자가 아닌 값이 있습니다.')
    records = []
    for metric, x in df.groupby('metric'):
        parts = sorted(x.part_id.unique(),key=str); operators = sorted(x.operator.unique(),key=str)
        repeats = sorted(x['repeat'].unique(),key=str)
        p, o, r = len(parts), len(operators), len(repeats)
        if p < 2 or o < 2 or r < 2 or len(x) != p*o*r:
            raise ValueError(f'{metric}: 균형 교차 Gage R&R에 2개 이상 부품, 작업자, 반복과 모든 조합이 필요합니다.')
        if (x.groupby(['part_id','operator'])['repeat'].nunique() != r).any():
            raise ValueError(f'{metric}: 반복 측정 수가 셀별로 다릅니다.')
        pivot = x.pivot_table(index='part_id',columns=['operator','repeat'],values='reading')
        if pivot.isna().any().any():
            raise ValueError(f'{metric}: 일부 교차 측정 조합이 없습니다.')
        y = np.empty((p, o, r),dtype=float)
        for i, part in enumerate(parts):
            for j, operator in enumerate(operators):
                for k, rep in enumerate(repeats):
                    y[i,j,k] = float(x.loc[(x.part_id == part)&(x.operator == operator)&(x['repeat']==rep),'reading'].iloc[0])
        grand = y.mean(); by_part = y.mean(axis=(1,2)); by_op = y.mean(axis=(0,2)); cell = y.mean(axis=2)
        ss_p = o*r * np.sum((by_part-grand)**2)
        ss_o = p*r * np.sum((by_op-grand)**2)
        ss_po = r * np.sum((cell - by_part[:,None] - by_op[None,:] + grand)**2)
        ss_e = np.sum((y - cell[:,:,None])**2)
        ms_p, ms_o = ss_p/(p-1), ss_o/(o-1)
        ms_po, ms_e = ss_po/((p-1)*(o-1)), ss_e/(p*o*(r-1))
        ev2 = max(ms_e,0.)
        interaction2 = max((ms_po-ms_e)/r,0.)
        operator2 = max((ms_o-ms_po)/(p*r),0.)
        part2 = max((ms_p-ms_po)/(o*r),0.)
        repeatability = float(np.sqrt(ev2)); reproducibility = float(np.sqrt(operator2+interaction2))
        grr = float(np.sqrt(ev2+operator2+interaction2))
        pt = (6*grr)/spec.tolerance(metric)
        grade = '양호 (참고 기준)' if pt <= 0.10 else ('조건부 (참고 기준)' if pt <= 0.30 else '부적합 (참고 기준)')
        records.append({'metric':metric, '부품 수':p, '작업자 수':o, '반복':r,
                        'EV 표준편차':repeatability, 'AV+상호작용 표준편차':reproducibility,
                        'Part 표준편차':float(np.sqrt(part2)), 'GRR 표준편차':grr,
                        'P/T Ratio':float(pt), 'P/T (%)':float(pt*100), '참고 판정':grade})
    return pd.DataFrame(records)


def msa_gate(golden: pd.DataFrame | None, grr: pd.DataFrame | None,
             max_bias_fraction: float = 0.10) -> tuple[str, str]:
    if golden is None or grr is None:
        return '미확인', 'Golden 및 Gauge R&R 데이터가 모두 있어야 계측 신뢰성을 확인할 수 있습니다.'
    if set(golden.metric) != {'vth','idsat'} or set(grr.metric) != {'vth','idsat'}:
        return '미확인', 'Vth와 Idsat에 대한 모든 MSA 자료가 필요합니다.'
    if (grr['P/T Ratio'] > 0.30).any():
        return '부적합', 'P/T가 30%를 넘는 항목이 있습니다. 재측정 또는 측정시스템 검토가 먼저입니다.'
    if (golden['|Bias|/허용폭 (%)'] > max_bias_fraction*100).any():
        return '주의', 'Golden Bias가 사용자 설정 경계값을 초과합니다. 보정 검토 후 진행하십시오.'
    if (grr['P/T Ratio'] > 0.10).any():
        return '주의', 'P/T가 10~30%인 항목이 있습니다. 활용 조건을 검토해야 합니다.'
    return '양호', '데모 기준 MSA 문턱을 만족했습니다. 원인은 추가 공정 증거가 있어야 확정할 수 있습니다.'
