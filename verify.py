# ============================================================
# 선정 종목 검증: 과거 스냅샷의 TOP 30이 이후 실제로 어땠는지 계산 -> data/verify.json
#
# 흐름:  load_snapshots -> download_prices -> evaluate(종목별) -> summarize(통계) -> 저장
# 검증 기준을 바꾸고 싶으면 아래 [설정]만 고치면 된다.
# ============================================================
import os, re, sys, json, glob, time
from datetime import datetime, timedelta
from statistics import mean, median
from zoneinfo import ZoneInfo
import pandas as pd

# ---------------- 설정 ----------------
SNAP_DIR = 'data/snapshots'
OUT = 'data/verify.json'
HORIZONS = (1, 3, 5, 10, 20)       # 선정일 기준 N거래일 후 종가로 평가
ENTRY_RULES = {                    # 시간대별 매수 가정
    '2100': 'next_open',           #  21시 선정 -> 다음 거래일 시가에 매수 (이미 지난 종가로는 살 수 없으므로)
    '1200': 'same_close',          #  12시 선정 -> 그날 종가에 매수
}                                  #  (signal = 스냅샷에 저장된 선정 시점 가격, same_close, next_open 중 선택)
COST_PCT = 0.25                    # 수수료·세금 가정(왕복 %), 종목 수익률에서만 차감
WINDOW_DAYS = 5                    # 목표가·손절가 도달 / 최대 하락폭을 볼 기간(거래일)
NLP_THRESHOLD = 30                 # 뉴스·공시 일치/엇갈림 기준 (index.html과 동일)
BENCH = {'KOSPI': '^KS11', 'KOSDAQ': '^KQ11'}   # 비교 지수
RECORD_LIMIT = 600                 # verify.json에 담을 최근 종목별 기록 수 (통계는 전체로 계산)
SIGNAL_LIMIT = 40                  # verify.json에 담을 최근 신호(날짜·시간대) 수

KST = ZoneInfo('Asia/Seoul')
NOW = datetime.now(KST)
FORCE_RUN = os.environ.get('FORCE_RUN') == '1'


# ---------------- 1. 스냅샷 읽기 ----------------
def load_nlp_snapshot(date, slot):
    """선정 당시의 감성 점수 {종목코드: 점수}. 없으면 빈 dict"""
    path = f'{SNAP_DIR}/{date}_{slot}_nlp.json'
    try:
        with open(path, encoding='utf-8') as f:
            return {c: v.get('score') for c, v in json.load(f).get('stocks', {}).items()}
    except Exception:
        return {}


def load_snapshots():
    """data/snapshots/날짜_슬롯.json -> 신호 목록 [{date, slot, stocks, nlp}]"""
    signals = []
    for path in sorted(glob.glob(f'{SNAP_DIR}/*.json')):
        m = re.match(r'^(\d{4}-\d{2}-\d{2})_(1200|2100)\.json$', os.path.basename(path))
        if not m:                               # *_nlp.json 등은 건너뜀
            continue
        try:
            with open(path, encoding='utf-8') as f:
                stocks = json.load(f).get('stocks', [])
        except Exception as e:
            print('[건너뜀]', os.path.basename(path), type(e).__name__)
            continue
        date, slot = m.groups()
        signals.append({'date': date, 'slot': slot, 'stocks': stocks, 'nlp': load_nlp_snapshot(date, slot)})
    return signals


# ---------------- 2. 가격 내려받기 ----------------
def _extract(raw, ticker):
    """yfinance 결과에서 한 종목의 open/high/low/close 표를 꺼낸다"""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                return None
            d = raw[ticker].copy()
        else:
            d = raw.copy()
        d.columns = [str(c).lower() for c in d.columns]
        d = d[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce').dropna(subset=['close'])
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d[~d.index.duplicated()]
    except Exception:
        return None


def download_prices(tickers, start, end, chunk=60):
    """{티커: 일봉 표}. 실패한 종목은 빠진다"""
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        raw = None
        for _ in range(3):
            try:
                raw = yf.download(part, start=start, end=end, auto_adjust=True, progress=False,
                                  group_by='ticker', threads=True)
                if raw is not None and not raw.empty:
                    break
            except Exception as e:
                print('  [다운로드 오류]', type(e).__name__)
            time.sleep(2)
        if raw is None or raw.empty:
            continue
        for t in part:
            df = _extract(raw, t)
            if df is not None and len(df):
                out[t] = df
    return out


# ---------------- 3. 종목별 평가 ----------------
def _ok(x):
    """NaN이 아닌 양수인가"""
    return x is not None and x == x and x > 0


def entry_point(df, sig_pos, rule, signal_price):
    """(진입 위치, 진입 가격). 진입할 수 없으면 (None, None)"""
    if rule == 'next_open':
        pos = sig_pos + 1
        price = df['open'].iloc[pos] if pos < len(df) else None
        return (pos, float(price)) if _ok(price) else (None, None)
    if rule == 'signal' and _ok(signal_price):
        return sig_pos, float(signal_price)
    price = df['close'].iloc[sig_pos]
    return (sig_pos, float(price)) if _ok(price) else (None, None)


def bench_price(bdf, date, field):
    """지수의 특정 날짜 가격. 없으면 None"""
    if bdf is None or date not in bdf.index:
        return None
    v = bdf[field].loc[date]
    return float(v) if _ok(v) else None


def window_result(df, start, entry, target, stop):
    """진입 후 WINDOW_DAYS 동안의 결과: ('target'|'stop'|'none'|'pending', 최대 하락폭 %)
    같은 날 둘 다 닿으면 보수적으로 손절로 본다. 목표·손절 가격은 선정 시점 기준 절대값."""
    if start is None:
        return 'pending', None
    end = start + WINDOW_DAYS
    if end > len(df):
        return 'pending', None
    outcome = 'none'
    for i in range(start, end):
        hi, lo = df['high'].iloc[i], df['low'].iloc[i]
        if _ok(stop) and lo <= stop:
            outcome = 'stop'
            break
        if _ok(target) and hi >= target:
            outcome = 'target'
            break
    mae = (df['low'].iloc[start:end].min() / entry - 1) * 100
    return outcome, round(float(mae), 2)


def nlp_group(score):
    if score is None:
        return '데이터 없음'
    if score >= NLP_THRESHOLD:
        return '일치'
    if score <= -NLP_THRESHOLD:
        return '엇갈림'
    return '중립'


def evaluate(sig, stock, prices):
    """한 신호(날짜·시간대)의 한 종목을 평가해 기록(dict)을 만든다. 가격 자료가 없으면 None"""
    ticker = stock.get('ticker')
    df, bdf = prices.get(ticker), prices.get(BENCH.get(stock.get('market')))
    date = pd.Timestamp(sig['date'])
    if df is None or date not in df.index:
        return None
    sig_pos = df.index.get_loc(date)
    rule = ENTRY_RULES.get(sig['slot'], 'next_open')
    entry_pos, entry = entry_point(df, sig_pos, rule, stock.get('price'))

    code = stock.get('code')
    nlp_score = sig['nlp'].get(code)
    rec = {'date': sig['date'], 'slot': sig['slot'], 'code': code, 'ticker': ticker, 'name': stock.get('name'),
           'market': stock.get('market'), 'rank': stock.get('rank'), 'score': stock.get('final_score'),
           'grade': stock.get('grade'), 'ma5_status': stock.get('ma5_status'),
           'nlp_score': nlp_score, 'nlp_group': nlp_group(nlp_score), 'entry_rule': rule,
           'entry': round(entry, 2) if entry else None, 'ret': {}, 'excess': {}, 'outcome': 'pending', 'mae': None}
    if entry_pos is None:
        return rec

    b_entry = bench_price(bdf, df.index[entry_pos], 'open' if rule == 'next_open' else 'close')
    for h in HORIZONS:
        exit_pos = sig_pos + h
        if exit_pos >= len(df):                       # 아직 N거래일이 지나지 않음
            continue
        ret = (float(df['close'].iloc[exit_pos]) / entry - 1) * 100 - COST_PCT
        rec['ret'][h] = round(ret, 2)
        b_exit = bench_price(bdf, df.index[exit_pos], 'close')
        if b_entry and b_exit:
            rec['excess'][h] = round(ret - (b_exit / b_entry - 1) * 100, 2)

    start = entry_pos if rule == 'next_open' else sig_pos + 1
    rec['outcome'], rec['mae'] = window_result(df, start, entry, stock.get('target_price'), stock.get('stop_price'))
    return rec


# ---------------- 4. 통계 ----------------
def stats(records, h):
    """기간 h의 평균·중앙값·승률·지수 대비. 표본이 없으면 None"""
    rets = [r['ret'][h] for r in records if h in r['ret']]
    exs = [r['excess'][h] for r in records if h in r['excess']]
    if not rets:
        return None
    return {'n': len(rets), 'avg': round(mean(rets), 2), 'median': round(median(rets), 2),
            'win': round(100 * sum(x > 0 for x in rets) / len(rets), 1),
            'avg_excess': round(mean(exs), 2) if exs else None,
            'beat': round(100 * sum(x > 0 for x in exs) / len(exs), 1) if exs else None}


def hit_stats(records):
    """목표가·손절가 도달 비율 (WINDOW_DAYS가 지난 기록만)"""
    done = [r for r in records if r['outcome'] != 'pending']
    if not done:
        return None
    n = len(done)
    return {'n': n, 'target': round(100 * sum(r['outcome'] == 'target' for r in done) / n, 1),
            'stop': round(100 * sum(r['outcome'] == 'stop' for r in done) / n, 1)}


def rank_group(r):
    return '1~10위' if r['rank'] <= 10 else '11~20위' if r['rank'] <= 20 else '21~30위'


def ma5_group(r):
    s = r.get('ma5_status')
    return '상승전환' if s == '상승전환' else '5일선 위(유지)' if s == '5일선 위' else '5일선 아래·하락전환'


# (키, 제목, 분류 함수, 표시 순서)  -> 비교 그룹을 늘리려면 여기에 한 줄 추가
GROUPS = [
    ('rank', '순위 구간', rank_group, ['1~10위', '11~20위', '21~30위']),
    ('ma5', '5일선 상태', ma5_group, ['상승전환', '5일선 위(유지)', '5일선 아래·하락전환']),
    ('nlp', '뉴스·공시', lambda r: r['nlp_group'], ['일치', '중립', '엇갈림', '데이터 없음']),
    ('slot', '선정 시간대', lambda r: '12시 장중' if r['slot'] == '1200' else '21시 종가', ['21시 종가', '12시 장중']),
]


def _row(label, rs):
    return {'label': label, 'records': len(rs), 'stats': {str(h): stats(rs, h) for h in HORIZONS}, 'hit': hit_stats(rs)}


def summarize(records):
    groups = []
    for key, title, fn, order in GROUPS:
        buckets = {}
        for r in records:
            buckets.setdefault(fn(r), []).append(r)
        groups.append({'key': key, 'title': title, 'rows': [_row(lb, buckets[lb]) for lb in order if lb in buckets]})
    return {'overall': _row('전체', records), 'groups': groups}


def per_signal(records):
    """신호(날짜·시간대)별 평균 수익률 / 초과수익 (최근 SIGNAL_LIMIT개)"""
    buckets = {}
    for r in records:
        buckets.setdefault((r['date'], r['slot']), []).append(r)
    out = []
    for (date, slot), rs in sorted(buckets.items(), reverse=True)[:SIGNAL_LIMIT]:
        item = {'date': date, 'slot': slot, 'n': len(rs), 'ret': {}, 'excess': {}}
        for h in HORIZONS:
            st = stats(rs, h)
            item['ret'][str(h)] = st['avg'] if st else None
            item['excess'][str(h)] = st['avg_excess'] if st else None
        out.append(item)
    return out


# ---------------- 5. 실행 ----------------
def main():
    if NOW.hour < 15 and not FORCE_RUN:
        print('[종료] 오후 3시 이전에는 당일 종가가 확정되지 않아 건너뜁니다 (강제 실행: FORCE_RUN=1)')
        return
    signals = load_snapshots()
    if not signals:
        print('[종료] 검증할 스냅샷(data/snapshots)이 없습니다')
        return

    tickers = sorted({s['ticker'] for sig in signals for s in sig['stocks'] if s.get('ticker')})
    start = (datetime.strptime(min(s['date'] for s in signals), '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
    end = (NOW + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f'[검증] 신호 {len(signals)}개, 종목 {len(tickers)}개, 기간 {start} ~ {end}')
    prices = download_prices(tickers + list(BENCH.values()), start, end)
    if not any(b in prices for b in BENCH.values()):
        print('[오류] 비교 지수 가격을 가져오지 못했습니다. 기존 verify.json을 유지합니다.')
        sys.exit(1)

    records = []
    for sig in signals:
        for stock in sig['stocks']:
            rec = evaluate(sig, stock, prices)
            if rec:
                records.append(rec)
    print(f'[검증] 평가한 기록 {len(records)}건 (가격 없는 종목 제외)')

    records.sort(key=lambda r: (r['date'], r['slot'], -r['rank']), reverse=True)
    shown = sorted(records[:RECORD_LIMIT], key=lambda r: (r['date'], r['slot'], r['rank']), reverse=True)
    payload = {
        'meta': {'generated_at': NOW.strftime('%Y-%m-%d %H:%M'), 'n_signals': len(signals), 'n_records': len(records),
                 'first_date': min(s['date'] for s in signals), 'last_date': max(s['date'] for s in signals),
                 'horizons': list(HORIZONS), 'entry_rules': ENTRY_RULES, 'cost_pct': COST_PCT,
                 'window_days': WINDOW_DAYS, 'nlp_threshold': NLP_THRESHOLD, 'bench': BENCH,
                 'records_shown': len(shown)},
        'summary': summarize(records),
        'signals': per_signal(records),
        'records': shown,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f'[저장] {OUT}')


if __name__ == '__main__':
    main()
