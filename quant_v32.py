# ============================================================
# QUANT V3.2 STABLE + RISK STRONG - GitHub Actions
# 12시(장중) / 21시(종가) 2회 실행 + JSON 저장 + 텔레그램 발송
# ============================================================

import sys, subprocess, importlib.util, warnings, os, time, json, html
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

for pkg, imp in [('yfinance','yfinance'),('pandas','pandas'),
                 ('numpy','numpy'),('tqdm','tqdm'),('requests','requests'),
                 ('scipy','scipy'),('scikit-learn','sklearn')]:
    if importlib.util.find_spec(imp) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from requests import Session

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)

# ============================================================
# [실행 시간대 설정] 한국시간 기준
#  - 15시 이전 실행 = 12시 슬롯(장중 시세), 이후 = 21시 슬롯(종가 기준)
#  - 수동 실행 시 환경변수 RUN_SLOT=1200 / 2100 으로 강제 지정 가능
#  - FORCE_RUN=1 이면 휴장일/데이터 미반영이어도 강제 실행
# ============================================================
KST = ZoneInfo('Asia/Seoul')
now_kst = datetime.now(KST)
RUN_SLOT = os.environ.get('RUN_SLOT') or ('1200' if now_kst.hour < 15 else '2100')
IS_INTRADAY = (RUN_SLOT == '1200')
SLOT_LABEL = '12시 장중 시세 (지연 가능)' if IS_INTRADAY else '21시 종가 기준'
FORCE_RUN = os.environ.get('FORCE_RUN') == '1'

# ============================================================
# [TEST SETTINGS]
# ============================================================
ANALYSIS_DATE = None  # None = 최신 거래일까지 자동 계산
KOSPI_COUNT = 200
KOSDAQ_COUNT = 200
LOOKBACK_DAYS = 420
MIN_HISTORY = 130

USE_PREVIOUS_SCORE = True
STABILITY_WEIGHT = 0.30

RISK_SCALE = 1.0
RISK_CAP = 25.0
HARD_RSI = 80
HARD_ATR_PCT = 10
HARD_RET5_UP = 30
HARD_RET1_DOWN = -8

# GitHub Actions 세션 설정
session = Session()
session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# 파일 저장 경로
BASE_DIR = '.'
JSON_DIR = 'data'
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(f'{JSON_DIR}/snapshots', exist_ok=True)

W_TAG = f'w{int(round(STABILITY_WEIGHT*100)):02d}'
R_TAG = 'riskstrong_fix'
D_TAG = str(ANALYSIS_DATE) if ANALYSIS_DATE else 'latest'

TOP10_FILE = f'{BASE_DIR}/quant_v3_v32_top10_{W_TAG}_{R_TAG}_{D_TAG}.csv'
TOP30_FILE = f'{BASE_DIR}/quant_v3_v32_top30_{W_TAG}_{R_TAG}_{D_TAG}.csv'
HISTORY_FILE = f'{BASE_DIR}/quant_v3_v32_history_{W_TAG}_{R_TAG}.csv'

CHUNK_SIZE = 75

def safe_float(x):
    try: return float(x)
    except: return np.nan

def pct_rank(s): return s.rank(pct=True) * 100

def get_ret(s, n):
    if len(s) <= n: return np.nan
    return (s.iloc[-1] / s.iloc[-1-n] - 1) * 100

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/p, adjust=False, min_periods=p).mean()
    al = l.ewm(alpha=1/p, adjust=False, min_periods=p).mean()
    return (100 - 100/(1 + ag/al.replace(0, np.nan))).fillna(50)

def atr(df, p=14):
    pc = df['close'].shift(1)
    tr = pd.concat([(df['high'] - df['low']), (df['high'] - pc).abs(), (df['low'] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False, min_periods=p).mean()

def macd(s):
    m = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m-sig

def grade(x):
    if x >= 85: return 'A+'
    if x >= 78: return 'A'
    if x >= 70: return 'B+'
    if x >= 62: return 'B'
    if x >= 55: return 'C'
    return 'D'

def ma5_status(r):
    """5일선 상태: 오늘/전일 종가와 5일선 비교"""
    up_now = r.price > r.ma5
    up_prev = r.prev_close > r.ma5_prev
    if up_now and not up_prev: return '상승전환'
    if (not up_now) and up_prev: return '하락전환'
    return '5일선 위' if up_now else '5일선 아래'

# ------------------------------------------------------------
# 한글 종목명 (캐시 data/names.json -> FinanceDataReader -> 네이버 순서로 시도, 모두 실패하면 영문명 유지)
# ------------------------------------------------------------
NAMES_FILE = f'{JSON_DIR}/names.json'

def load_names():
    try:
        with open(NAMES_FILE, encoding='utf-8') as f: return json.load(f)
    except: return {}

def fetch_korean_names(codes, cache):
    missing = [c for c in codes if c not in cache]
    if not missing: return cache
    # 1) FinanceDataReader 상장 종목 목록
    try:
        if importlib.util.find_spec('FinanceDataReader') is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'finance-datareader'])
        import FinanceDataReader as fdr
        lst = fdr.StockListing('KRX')
        m = dict(zip(lst['Code'].astype(str).str.zfill(6), lst['Name']))
        for c in missing:
            if m.get(c): cache[c] = str(m[c])
    except Exception as e:
        print('[한글명] FinanceDataReader 실패:', e)
    # 2) 네이버 모바일 API (남은 종목만)
    for c in [c for c in missing if c not in cache]:
        try:
            r = requests.get(f'https://m.stock.naver.com/api/stock/{c}/basic',
                             headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            nm = r.json().get('stockName') if r.status_code == 200 else None
            if nm: cache[c] = nm
        except Exception as e:
            print(f'[한글명] 네이버 조회 실패 {c}:', e)
        time.sleep(0.3)
    left = [c for c in missing if c not in cache]
    print(f'[한글명] 신규 {len(missing)}개 중 {len(missing)-len(left)}개 성공' + (f', 실패(영문명 유지): {left}' if left else ''))
    return cache

# ------------------------------------------------------------
# 동적 유니버스 수집 (yf.screen)
# ------------------------------------------------------------
def get_top(exchange, market, n):
    from yfinance import EquityQuery
    q = EquityQuery('and', [EquityQuery('eq', ['region', 'kr']), EquityQuery('eq', ['exchange', exchange])])
    print(f'{market} 상위 {n}개 종목 스크리닝 조회...')
    try:
        r = yf.screen(q, offset=0, size=n, sortField='intradaymarketcap', sortAsc=False)
    except Exception as e:
        print(f'[{market} 스크리닝 오류]', e)
        return pd.DataFrame()

    rows = []
    for x in r.get('quotes', []):
        t = x.get('symbol')
        if not t or not (t.endswith('.KS') or t.endswith('.KQ')): continue
        rows.append({
            'ticker': t,
            'name': x.get('shortName') or x.get('longName') or t,
            'market': 'KOSPI' if t.endswith('.KS') else 'KOSDAQ',
            'market_cap': safe_float(x.get('intradayMarketCap', x.get('marketCap', np.nan)))
        })
    df = pd.DataFrame(rows).drop_duplicates('ticker')
    if df.empty: return df
    return df.sort_values('market_cap', ascending=False, na_position='last').head(n).reset_index(drop=True)

def get_close(data, t):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if t in data.columns.get_level_values(0): return pd.to_numeric(data[t]['Close'], errors='coerce').dropna()
            return pd.to_numeric(data['Close'][t], errors='coerce').dropna()
        return pd.to_numeric(data['Close'], errors='coerce').dropna()
    except: return pd.Series(dtype=float)

def get_stock(raw, t):
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0): return None
            d = raw[t].copy()
        else: d = raw.copy()
        d.columns = [str(c).lower() for c in d.columns]
        need = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in d.columns for c in need): return None
        for c in need: d[c] = pd.to_numeric(d[c], errors='coerce')
        return d.dropna(subset=['close'])
    except: return None

def load_previous_scores(current_analysis_date):
    if not USE_PREVIOUS_SCORE or not os.path.exists(HISTORY_FILE): return {}, {}
    try:
        hist = pd.read_csv(HISTORY_FILE)
        hist['analysis_date'] = pd.to_datetime(hist['analysis_date'], errors='coerce')
        hist['final_score'] = pd.to_numeric(hist['final_score'], errors='coerce')
        hist['rank'] = pd.to_numeric(hist['rank'], errors='coerce')
        current_date = pd.Timestamp(current_analysis_date)
        hist = hist[hist['analysis_date'].notna() & hist['final_score'].notna() & (hist['analysis_date'] < current_date)].copy()
        if hist.empty: return {}, {}
        latest_date = hist['analysis_date'].max()
        prev = hist[hist['analysis_date'] == latest_date].copy()
        return dict(zip(prev['ticker'].astype(str), prev['final_score'].astype(float))), dict(zip(prev['ticker'].astype(str), prev['rank'].astype(float)))
    except: return {}, {}

print('=' * 100)
print(f'QUANT V3.2 STABLE + RISK STRONG 시작 | 슬롯: {SLOT_LABEL} | KST {now_kst:%Y-%m-%d %H:%M}')
print('=' * 100)

kospi = get_top('KSC', 'KOSPI', KOSPI_COUNT)
kosdaq = get_top('KOE', 'KOSDAQ', KOSDAQ_COUNT)
universe = pd.concat([kospi, kosdaq], ignore_index=True).drop_duplicates('ticker').reset_index(drop=True)

if universe.empty or len(universe) < 50:
    print("[치명적 오류] 유니버스 종목을 불러오지 못했습니다.")
    sys.exit(1)

if ANALYSIS_DATE:
    ad = pd.Timestamp(ANALYSIS_DATE)
    end = ad + pd.Timedelta(days=1)
else:
    ad = None
    end = pd.Timestamp(now_kst.date()) + pd.Timedelta(days=1)   # 한국 날짜 기준

start = end - pd.Timedelta(days=LOOKBACK_DAYS)

def download_index(sym, tries=4):
    for k in range(tries):
        try:
            x = yf.download(sym, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True, progress=False, threads=False, session=session)
            ser = get_close(x, sym)
            if len(ser) >= 30: return ser
        except: pass
        time.sleep(3 * (k + 1))
    return pd.Series(dtype=float)

ksp = download_index('^KS11')
kqd = download_index('^KQ11')
if len(ksp) < 30 or len(kqd) < 30:
    print("[치명적 오류] 지수 데이터 수집 실패")
    sys.exit(1)

# 휴장일/데이터 미반영 체크: 오늘 날짜 데이터가 없으면 발송·저장 없이 종료
if ad is None and not FORCE_RUN and ksp.index[-1].date() != now_kst.date():
    print(f'[종료] 오늘({now_kst.date()}) 데이터 없음 (최신: {ksp.index[-1].date()}). 휴장일이거나 데이터 미반영입니다. 강제 실행은 FORCE_RUN=1')
    sys.exit(0)

mr = {'KOSPI': {n: get_ret(ksp, n) for n in [1, 3, 5, 10, 20]}, 'KOSDAQ': {n: get_ret(kqd, n) for n in [1, 3, 5, 10, 20]}}

def download_in_chunks(tickers, chunk_size=75):
    frames = []
    total = len(tickers)
    for i in range(0, total, chunk_size):
        chunk = tickers[i:i+chunk_size]
        for _ in range(3):
            try:
                x = yf.download(chunk, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True, repair=True, progress=False, threads=True, session=session, group_by='ticker')
                if x is not None and not x.empty:
                    frames.append(x)
                    break
            except: pass
            time.sleep(1)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]

raw = download_in_chunks(universe.ticker.tolist(), chunk_size=CHUNK_SIZE)

records = []
for _, st in tqdm(universe.iterrows(), total=len(universe)):
    t, n, m = st.ticker, st['name'], st.market
    d = get_stock(raw, t)
    if d is None: continue
    if ad is not None: d = d[d.index <= ad]
    if len(d) < MIN_HISTORY: continue
    c, v = d.close, d.volume
    price = float(c.iloc[-1])
    ma5_s = c.rolling(5).mean()
    ma5, ma20, ma60, ma120 = ma5_s.iloc[-1], c.rolling(20).mean().iloc[-1], c.rolling(60).mean().iloc[-1], c.rolling(120).mean().iloc[-1]
    ma5_prev, prev_close = float(ma5_s.iloc[-2]), float(c.iloc[-2])   # 5일선 상승전환 판정용
    r, a = float(rsi(c).iloc[-1]), float(atr(d).iloc[-1])
    ap = a / price * 100 if price > 0 else np.nan
    ml, ms, mh = macd(c)
    rets = {x: get_ret(c, x) for x in [1, 3, 5, 10, 20]}
    vr = v.tail(5).mean() / v.tail(20).mean() if v.tail(20).mean() > 0 else np.nan
    tvr = (c * v).tail(5).mean() / (c * v).tail(20).mean() if (c * v).tail(20).mean() > 0 else np.nan
    h20, h52 = c.tail(20).max(), c.tail(252).max()
    hd20 = price / h20 * 100 if h20 > 0 else np.nan
    records.append({
        'ticker': t, 'name': n, 'market': m, 'analysis_date': d.index[-1].strftime('%Y-%m-%d'), 'price': price,
        'prev_close': prev_close, 'ma5_prev': ma5_prev,
        'ret1': rets[1], 'ret3': rets[3], 'ret5': rets[5], 'ret10': rets[10], 'ret20': rets[20],
        'rsi14': r, 'ma5': ma5, 'ma20': ma20, 'ma60': ma60, 'ma120': ma120,
        'macd': float(ml.iloc[-1]), 'macd_signal': float(ms.iloc[-1]), 'macd_hist': float(mh.iloc[-1]),
        'volume_ratio': vr, 'tv_ratio': tvr, 'atr14': a, 'atr_pct': ap,
        'volatility20': c.pct_change().tail(20).std() * 100,
        'high20_distance': hd20, 'high52_distance': price / h52 * 100 if h52 > 0 else np.nan,
        'rs1': rets[1] - mr[m][1], 'rs5': rets[5] - mr[m][5], 'rs20': rets[20] - mr[m][20],
        'ma_alignment': int(price > ma20) + int(ma20 > ma60) + int(ma60 > ma120),
        'breakout20': int(len(c) >= 22 and price >= c.iloc[-21:-1].max())
    })

features = pd.DataFrame(records)
for c in ['price','prev_close','ma5_prev','ret1','ret3','ret5','ret10','ret20','rsi14','ma5','ma20','ma60','ma120','macd','macd_signal','macd_hist','volume_ratio','tv_ratio','atr14','atr_pct','volatility20','high20_distance','high52_distance','rs1','rs5','rs20']:
    features[c] = pd.to_numeric(features[c], errors='coerce')
features = features.dropna(subset=['price','ma5','ma20','ma60','ma120','rsi14','atr14','ret5','ret20','rs5','rs20']).copy()

# Score Calculation
features['momentum_score'] = (pct_rank(features.ret1) * .15 + pct_rank(features.ret3) * .20 + pct_rank(features.ret5) * .30 + pct_rank(features.ret20) * .35) * .25
features['relative_strength_score'] = (pct_rank(features.rs1) * .20 + pct_rank(features.rs5) * .45 + pct_rank(features.rs20) * .35) * .15
ma2060, ma60120 = features.ma20 / features.ma60 - 1, features.ma60 / features.ma120 - 1
features['trend_score'] = (features.ma_alignment / 3 * 60 + pct_rank(ma2060) * .25 + pct_rank(ma60120) * .15).clip(0, 100) * .15
features['acceleration_score'] = pct_rank(features.ret1 * .35 + features.ret3 * .35 + features.ret5 * .30) * .10
features['flow_score'] = pct_rank(features.volume_ratio.clip(0,4) * .5 + features.tv_ratio.clip(0,4) * .5) * .10
features['technical_score'] = pd.Series(np.where((features.rsi14 >= 50) & (features.rsi14 <= 68), 4, 0) + np.where(features.macd_hist > 0, 3, 0) + np.where(features.breakout20 == 1, 3, 0), index=features.index).clip(0,10)

# Risk Score (강화)
ma20_gap = (features.price / features.ma20 - 1) * 100
ma5_gap = (features.price / features.ma5 - 1) * 100
avg_pace = features.rs5 / 5.0
risk = np.zeros(len(features))
risk += np.where(features.rsi14 >= 72, 8, np.where(features.rsi14 >= 68, 5, np.where(features.rsi14 >= 65, 2, 0)))
risk += np.where(features.high20_distance >= 99, 4, np.where(features.high20_distance >= 96, 2.5, 0))
risk += np.where(features.ret5 >= 15, 4, np.where(features.ret5 >= 10, 2.5, 0))
risk += np.where(features.atr_pct >= 6, 4, np.where(features.atr_pct >= 4.5, 2.5, 0))
risk += np.where(features.ret1 <= -6, 5, np.where(features.ret1 <= -3, 2.5, 0))
risk += np.where(features.price < features.ma20, 4, 0)
risk += np.where(features.ret1 >= 8, 3, np.where(features.ret1 >= 5, 1.5, 0))
risk += np.where(ma20_gap >= 20, 4, np.where(ma20_gap >= 12, 2, 0))
risk += np.where(ma5_gap >= 6, 2, 0)
risk += np.where((features.rs5 > 0) & (features.rs1 < avg_pace * 0.3), 2.5, 0)
risk += np.where(features.macd_hist < 0, 1.5, 0)
features['risk_score'] = pd.Series(risk * RISK_SCALE, index=features.index).clip(0, RISK_CAP)

features['base_score'] = features.momentum_score + features.relative_strength_score + features.trend_score + features.acceleration_score + features.flow_score + features.technical_score
current_analysis_date = pd.to_datetime(features['analysis_date']).max()
prev_score_map, prev_rank_map = load_previous_scores(current_analysis_date)
features['prev_score'] = features['ticker'].map(prev_score_map)
features['prev_rank'] = features['ticker'].map(prev_rank_map)

features['score_before_stability'] = (features.base_score - features.risk_score).clip(0,100)
features['stability_score'] = features['score_before_stability']
if USE_PREVIOUS_SCORE and STABILITY_WEIGHT > 0:
    has_prev = features['prev_score'].notna()
    features.loc[has_prev, 'stability_score'] = features.loc[has_prev, 'score_before_stability'] + (features.loc[has_prev, 'prev_score'] - features.loc[has_prev, 'score_before_stability']) * STABILITY_WEIGHT

features['final_score'] = features.stability_score.clip(0,100)
features['hard_filter'] = (features.rsi14 >= HARD_RSI) | (features.atr_pct >= HARD_ATR_PCT) | (features.ret5 >= HARD_RET5_UP) | ((features.ret1 <= HARD_RET1_DOWN) & (features.ret5 <= -3))

# ATR 기반 동적 목표가 및 기대수익률
features['target_price'] = features.price + features.atr14 * (.45 + features.final_score / 100 * .25)
features['stop_price'] = np.maximum(features.price - features.atr14 * 1.40, features.price * .88)
features['target_return'] = (features.target_price / features.price - 1) * 100
features['stop_return'] = (features.stop_price / features.price - 1) * 100

eligible = features[~features.hard_filter].sort_values(['final_score', 'rs20', 'ret5', 'tv_ratio'], ascending=[False, False, False, False]).reset_index(drop=True)
eligible['rank'] = np.arange(1, len(eligible)+1)
top30 = eligible.head(30).copy()
top10 = eligible.head(10).copy()

for df in (top10, top30):
    df['grade'] = df.final_score.apply(grade)
    df['code'] = df.ticker.str.split('.').str[0]
    df['ma5_gap'] = (df.price / df.ma5 - 1) * 100
    df['ma20_gap'] = (df.price / df.ma20 - 1) * 100
    df['ma5_status'] = df.apply(ma5_status, axis=1)
    df['rank_change'] = df['prev_rank'] - df['rank']   # 양수 = 순위 상승

# 한글 종목명 적용 (원래 이름은 name_en에 보관)
names_cache = fetch_korean_names(sorted(set(top30['code']) | set(top10['code'])), load_names())
with open(NAMES_FILE, 'w', encoding='utf-8') as f:
    json.dump(names_cache, f, ensure_ascii=False, indent=1, sort_keys=True)
for df in (top10, top30):
    df['name_en'] = df['name']
    df['name'] = df['code'].map(names_cache).fillna(df['name'])

top10.to_csv(TOP10_FILE, index=False, encoding='utf-8-sig')
top30.to_csv(TOP30_FILE, index=False, encoding='utf-8-sig')

# 히스토리 누적: 안정성 점수 계산이 종가 기준으로만 이어지도록 21시(종가) 슬롯에서만 저장
if not IS_INTRADAY:
    new_hist = top30.copy()
    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE)
        old = old[~old['analysis_date'].astype(str).isin(new_hist['analysis_date'].astype(str).unique())]
        all_hist = pd.concat([old, new_hist], ignore_index=True)
    else:
        all_hist = new_hist.copy()
    all_hist.drop_duplicates(subset=['analysis_date', 'ticker'], keep='last').to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')

# ------------------------------------------------------------
# JSON 저장 (정적 웹페이지가 읽는 파일)
# ------------------------------------------------------------
date_disp = current_analysis_date.strftime('%Y-%m-%d')
json_cols = ['rank','rank_change','prev_rank','code','ticker','name','name_en','market','price','ret1','ret5','ret20',
             'ma5','ma20','ma60','ma120','ma5_gap','ma20_gap','ma5_status','rsi14','macd_hist',
             'volume_ratio','atr_pct','risk_score','final_score','grade',
             'target_price','target_return','stop_price','stop_return']
payload = {
    'meta': {
        'generated_at': now_kst.strftime('%Y-%m-%d %H:%M'),
        'slot': RUN_SLOT,
        'slot_label': SLOT_LABEL,
        'is_intraday': IS_INTRADAY,
        'analysis_date': date_disp,
        'universe_size': int(len(features)),
        'strategy': 'QUANT V3.2 STABLE + RISK STRONG',
    },
    'stocks': json.loads(top30[json_cols].round(2).to_json(orient='records', force_ascii=False)),
}
for path in (f'{JSON_DIR}/latest.json', f'{JSON_DIR}/snapshots/{date_disp}_{RUN_SLOT}.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
print(f'[저장] {JSON_DIR}/latest.json, snapshots/{date_disp}_{RUN_SLOT}.json')

# ------------------------------------------------------------
# 텔레그램 발송 (4096자 제한 때문에 나눠서 전송)
# ------------------------------------------------------------
def fmt_row(r):
    chg = r['rank_change']
    if pd.isna(chg): mv = ''
    elif chg > 0: mv = f' ▲{int(chg)}'
    elif chg < 0: mv = f' ▼{int(-chg)}'
    else: mv = ' -'
    nm = html.escape(str(r['name']))
    return (f"<b>{int(r['rank'])}위{mv}: {nm} ({r['code']})</b>\n"
            f"  • 점수 {r['final_score']:.1f} ({r['grade']}) | 위험 {r['risk_score']:.1f}\n"
            f"  • 현재가 {r['price']:,.0f}원 | 5일선 {r['ma5']:,.0f}원 ({r['ma5_gap']:+.1f}%) {r['ma5_status']}\n"
            f"  • 목표가 {r['target_price']:,.0f}원 ({r['target_return']:+.2f}%) | RSI {r['rsi14']:.0f}\n\n")

def send_telegram(header, rows, footer):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not (token and chat_id):
        print("[알림] 텔레그램 토큰/채팅ID 없음 - 발송 생략")
        return
    chunks, cur = [], header
    for r in rows:
        if len(cur) + len(r) > 3800:
            chunks.append(cur); cur = ''
        cur += r
    chunks.append(cur + footer)
    for i, text in enumerate(chunks, 1):
        try:
            res = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=30)
            print(f"[알림] 텔레그램 {i}/{len(chunks)} 전송", "성공" if res.status_code == 200 else f"실패: {res.text}")
        except Exception as e:
            print("[알림 오류] 텔레그램 예외:", e)
        time.sleep(1)

header = f"📊 <b>[QUANT V3.2 TOP 30] {SLOT_LABEL}</b>\n기준일 {date_disp} | 발송 {now_kst:%H:%M}\n\n"
footer = "※ 투자 권유가 아닌 참고용 정보입니다."
if IS_INTRADAY:
    footer = "※ 12시 데이터는 장중 시세라 지연될 수 있고, 거래량 지표는 장중 누적 기준입니다.\n" + footer
send_telegram(header, [fmt_row(r) for _, r in top30.iterrows()], footer)

print("모든 프로세스 완료.")
