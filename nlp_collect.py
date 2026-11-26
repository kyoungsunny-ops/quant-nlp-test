# ============================================================
# 뉴스·공시 수집 + 규칙 기반 감성 분석 -> data/nlp.json
#  - 입력: data/latest.json (퀀트 TOP 30)
#  - 공시: OpenDART API (환경변수 DART_API_KEY 필요)
#  - 뉴스: 구글 뉴스 RSS (키 불필요, 제목·링크·언론사만 사용)
#  - 감성: 키워드 규칙 (숫자는 코드가 계산, LLM 미사용)
# ============================================================
import os, sys, io, json, html, time, zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote
import requests

KST = ZoneInfo('Asia/Seoul')
NOW = datetime.now(KST)
LATEST = 'data/latest.json'
OUT = 'data/nlp.json'
DART_KEY = os.environ.get('DART_API_KEY', '').strip()
FORCE_RUN = os.environ.get('FORCE_RUN') == '1'

DART_DAYS = 14      # 공시 조회 기간(일)
NEWS_DAYS = 3       # 뉴스 조회 기간(일)
MAX_DART = 3        # 종목당 표시할 공시 수
MAX_NEWS = 4        # 종목당 표시할 뉴스 수
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'}

# ------------------------------------------------------------
# 키워드 규칙: (키워드, 가중치, 이슈 태그)  + 가중치는 항목당 -2 ~ +2로 제한
# ------------------------------------------------------------
RULES = [
    ('공급계약', 2, '수주'), ('수주', 2, '수주'), ('단일판매', 2, '수주'),
    ('흑자전환', 2, '실적 개선'), ('사상 최대', 2, '실적 개선'), ('최대 실적', 2, '실적 개선'),
    ('어닝 서프라이즈', 2, '실적 개선'), ('실적 개선', 1, '실적 개선'), ('호실적', 1, '실적 개선'),
    ('영업이익 증가', 1, '실적 개선'), ('적자 탈출', 1, '실적 개선'), ('상회', 1, '실적 개선'),
    ('목표주가 상향', 1, '목표가 상향'), ('목표가 상향', 1, '목표가 상향'), ('투자의견 상향', 1, '목표가 상향'),
    ('자기주식 취득', 1, '자사주'), ('자사주 매입', 1, '자사주'), ('자기주식취득', 1, '자사주'),
    ('자기주식 소각', 2, '자사주'), ('자사주 소각', 2, '자사주'), ('주식소각', 2, '자사주'),
    ('배당결정', 1, '배당'), ('배당 확대', 1, '배당'), ('임상 성공', 2, '임상'), ('허가 획득', 1, '허가'),
    ('신고가', 1, '신고가'), ('증설', 1, '증설'),
    ('유상증자', -2, '유상증자'), ('전환사채', -1, '자금조달'), ('신주인수권부사채', -1, '자금조달'), ('교환사채', -1, '자금조달'),
    ('적자전환', -2, '실적 부진'), ('어닝 쇼크', -2, '실적 부진'), ('영업손실', -1, '실적 부진'),
    ('적자', -1, '실적 부진'), ('실적 부진', -1, '실적 부진'), ('하회', -1, '실적 부진'),
    ('목표주가 하향', -1, '목표가 하향'), ('목표가 하향', -1, '목표가 하향'), ('투자의견 하향', -1, '목표가 하향'),
    ('소송', -1, '소송'), ('횡령', -2, '횡령·배임'), ('배임', -2, '횡령·배임'),
    ('거래정지', -2, '거래정지'), ('관리종목', -2, '관리종목'), ('상장폐지', -2, '상장폐지'), ('의견거절', -2, '감사의견'),
    ('불성실공시', -1, '불성실공시'), ('임상 실패', -2, '임상 지연'), ('임상 중단', -2, '임상 지연'), ('임상 지연', -1, '임상 지연'),
    ('리콜', -1, '리콜'), ('과징금', -1, '제재'), ('제재', -1, '제재'), ('영업정지', -2, '영업정지'),
    ('감자결정', -1, '감자'), ('급락', -1, '급락'), ('블록딜', -1, '지분 매각'),
    ('투자경고', -1, '시장조치'), ('투자위험', -2, '시장조치'), ('투자주의', -1, '시장조치'),
    ('부도', -2, '부도'), ('파산', -2, '파산'), ('회생절차', -2, '회생'),
]
EXCLUDE = {'적자': ['탈출', '축소', '개선'], '소송': ['승소', '취하']}   # 이런 말이 같이 있으면 해당 키워드는 무시
DART_NOISE = ('소유상황보고서', '대량보유상황보고서')                    # 너무 잦아 의미가 적은 공시 제외

# 뒤집기 규칙: 키워드 바로 뒤에 아래 단어가 붙으면 그 키워드의 점수 부호를 뒤집는다
#   "실적 부진 선반영" -> 긍정, "유상증자 철회" -> 긍정, "수주 무산" -> 부정
#   'full' = 점수를 그대로 뒤집음 / 'soft' = 뒤집되 크기를 +-FLIP_MAX로 제한(영향이 작은 경우)
FLIP_RULES = (('선반영', 'soft'), ('둔화', 'soft'), ('우려 완화', 'soft'),
              ('철회', 'full'), ('무산', 'full'), ('취소', 'full'), ('해소', 'full'))
FLIP_MAX = 1          # 'soft' 뒤집기의 최대 점수 크기
FLIP_GAP = 12         # 키워드와 뒤집기 단어 사이 허용 글자 수
DELIMS = set(',，·…‘’“”"\'!?.;:()[]「」』『|/')   # 이런 기호가 사이에 있으면 서로 다른 문구로 보고 뒤집지 않음


def _flip_mode(text, start, end):
    """키워드(start~end) 바로 뒤에 뒤집기 단어가 붙었거나('실적 부진 선반영'),
    '철회된 유상증자'처럼 바로 앞에 '단어+된/한'으로 붙어 있으면 (단어, 방식)을, 아니면 None을 반환"""
    tail = text[end:end + FLIP_GAP + 6]
    for word, mode in FLIP_RULES:
        i = tail.find(word)
        if i != -1 and i <= FLIP_GAP and not any(c in DELIMS for c in tail[:i]):
            return word, mode
    head = text[max(0, start - 10):start]
    for word, mode in FLIP_RULES:
        j = head.rfind(word)
        if j != -1:
            rest = head[j + len(word):]
            if len(rest.strip()) <= 3 and rest.strip().startswith(('된', '한', '되는', '하는')) \
                    and not any(c in DELIMS for c in rest):
                return word, mode
    return None


def classify(text):
    """텍스트 -> (가중치 -2~+2, 이슈 태그 목록)"""
    w, soft_w, tags = 0, 0, []
    for kw, wt, tag in RULES:
        i = text.find(kw)
        if i == -1:
            continue
        if kw in EXCLUDE and any(x in text for x in EXCLUDE[kw]):
            continue
        t = tag
        fm = _flip_mode(text, i, i + len(kw))
        if fm:
            word, mode = fm
            if mode == 'full':
                w += -wt
            else:
                soft_w += -wt        # soft 항목은 따로 모아 합계를 제한 ('적자전환'+'적자' 중복 방지)
            t = f'{tag} {word}'
        else:
            w += wt
        if t not in tags:
            tags.append(t)
    w += max(-FLIP_MAX, min(FLIP_MAX, soft_w))
    return max(-2, min(2, w)), tags


def label(w):
    return 'pos' if w > 0 else 'neg' if w < 0 else 'neu'


def safe_get(url, **kw):
    """오류 메시지에 URL(키 포함)이 노출되지 않도록 예외 내용을 가린다"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=kw.pop('timeout', 20), **kw)
        if r.status_code != 200:
            print(f'  [HTTP {r.status_code}] {url.split("?")[0]}')
            return None
        return r
    except Exception as e:
        print(f'  [요청 실패] {type(e).__name__} {url.split("?")[0]}')
        return None


def load_corp_map():
    """종목코드(6자리) -> DART 고유번호(8자리)"""
    r = safe_get('https://opendart.fss.or.kr/api/corpCode.xml', params={'crtfc_key': DART_KEY}, timeout=60)
    if r is None:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
    except zipfile.BadZipFile:
        print('  [DART] 고유번호 응답이 ZIP이 아닙니다 (키 오류 또는 한도 초과 가능):', r.text[:120].replace(DART_KEY, '***'))
        return None
    root = ET.fromstring(z.read(z.namelist()[0]))
    m = {}
    for it in root.iter('list'):
        sc = (it.findtext('stock_code') or '').strip()
        cc = (it.findtext('corp_code') or '').strip()
        if sc and cc:
            m[sc] = cc
    return m


def fetch_dart(corp_code):
    """최근 공시 목록. 실패 None, 데이터 없음 []"""
    p = {'crtfc_key': DART_KEY, 'corp_code': corp_code,
         'bgn_de': (NOW - timedelta(days=DART_DAYS)).strftime('%Y%m%d'), 'end_de': NOW.strftime('%Y%m%d'),
         'page_count': 30, 'sort': 'date', 'sort_mth': 'desc'}
    r = safe_get('https://opendart.fss.or.kr/api/list.json', params=p)
    if r is None:
        return None
    j = r.json()
    if j.get('status') == '013':
        return []
    if j.get('status') != '000':
        print(f'  [DART status {j.get("status")}] {j.get("message")}')
        return None
    items = []
    for it in j.get('list', []):
        nm = (it.get('report_nm') or '').strip()
        if not nm or any(n in nm for n in DART_NOISE):
            continue
        dt = it.get('rcept_dt') or ''
        if len(dt) != 8:
            continue
        w, tags = classify(nm)
        items.append({'k': '공시', 't': nm, 'd': '', 'u': f'https://dart.fss.or.kr/dsaf001/main.do?rcpNo={it.get("rcept_no")}',
                      'l': label(w), 'dt': f'{dt[4:6]}-{dt[6:8]}', 'ts': dt, 'w': w, 'tags': tags, 'mult': 1.5})
    return items


def fetch_news(name):
    """구글 뉴스 RSS (제목·링크·언론사만). 실패 None"""
    q = f'"{name}" when:{NEWS_DAYS}d'
    r = safe_get('https://news.google.com/rss/search?q=' + quote(q) + '&hl=ko&gl=KR&ceid=KR:ko')
    if r is None:
        return None
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return None
    items, seen = [], set()
    cutoff = NOW - timedelta(days=NEWS_DAYS)
    for it in root.iter('item'):
        title = html.unescape(it.findtext('title') or '').strip()
        src = (it.findtext('source') or '').strip()
        if src and title.endswith(' - ' + src):
            title = title[:-(len(src) + 3)]
        try:
            pub = parsedate_to_datetime(it.findtext('pubDate')).astimezone(KST)
        except Exception:
            continue
        key = title[:25]
        if not title or key in seen or pub < cutoff:
            continue
        seen.add(key)
        w, tags = classify(title)
        items.append({'k': '뉴스', 't': title, 'd': src, 'u': (it.findtext('link') or '').strip(),
                      'l': label(w), 'dt': pub.strftime('%m-%d'), 'ts': pub.strftime('%Y%m%d%H%M'), 'w': w, 'tags': tags, 'mult': 1.0})
    return items


def pick(items, n):
    """중요도(가중치 절댓값) 높은 순으로 n개 고른 뒤 최신순 정렬"""
    top = sorted(items, key=lambda x: (abs(x['w']), x['ts']), reverse=True)[:n]
    return sorted(top, key=lambda x: x['ts'], reverse=True)


def build_stock(dart, news):
    chosen = pick(dart or [], MAX_DART) + pick(news or [], MAX_NEWS)
    raw = sum(x['w'] * x['mult'] for x in chosen)
    score = max(-100, min(100, round(raw * 25)))
    tags = []
    for x in sorted(chosen, key=lambda x: -abs(x['w'])):
        for t in x['tags']:
            if t not in tags:
                tags.append(t)
    news_out = [{k: x[k] for k in ('k', 't', 'd', 'u', 'l', 'dt')} for x in chosen]
    return {'score': score, 'issues': tags[:4], 'news': news_out}


def save_snapshot(payload, latest_meta):
    """검증 페이지용: 선정 당시의 감성 점수를 날짜·시간대별로 보관 (점수·태그만 저장해 용량을 줄임)"""
    date, slot = latest_meta.get('analysis_date'), latest_meta.get('slot')
    if not date or not slot:
        print('[스냅샷] latest.json에 analysis_date/slot이 없어 건너뜁니다')
        return
    os.makedirs('data/snapshots', exist_ok=True)
    small = {'meta': payload['meta'],
             'stocks': {c: {'score': v['score'], 'issues': v['issues']} for c, v in payload['stocks'].items()}}
    path = f'data/snapshots/{date}_{slot}_nlp.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(small, f, ensure_ascii=False)
    print(f'[스냅샷] {path}')


def main():
    if not os.path.exists(LATEST):
        print('[오류] data/latest.json 이 없습니다. 퀀트 단계를 먼저 실행하세요.')
        sys.exit(1)
    with open(LATEST, encoding='utf-8') as f:
        d = json.load(f)
    gen = (d.get('meta', {}).get('generated_at') or '')[:10]
    if not FORCE_RUN and gen != NOW.strftime('%Y-%m-%d'):
        print(f'[종료] latest.json 이 오늘 생성된 것이 아닙니다({gen}). 휴장일로 보고 건너뜁니다. 강제 실행은 FORCE_RUN=1')
        return

    corp = None
    if DART_KEY:
        corp = load_corp_map()
        print(f'[DART] 고유번호 {"불러오기 실패" if corp is None else str(len(corp)) + "개 로드"}')
    else:
        print('[DART] DART_API_KEY 없음 - 공시 수집 생략')

    out, ok = {}, 0
    for s in d['stocks']:
        code, name = s['code'], s['name']
        dart = fetch_dart(corp[code]) if corp and code in corp else None
        time.sleep(0.2)
        news = fetch_news(name)
        time.sleep(0.5)
        if dart is not None or news is not None:
            ok += 1
        out[code] = build_stock(dart, news)
        print(f'{s["rank"]:>2}위 {name} | 공시 {len(dart) if dart is not None else "미조회"} 뉴스 {len(news) if news is not None else "실패"}'
              f' -> 감성 {out[code]["score"]:+d} {out[code]["issues"]}')

    if ok == 0:
        print('[오류] 모든 종목의 수집이 실패했습니다. 기존 nlp.json 을 유지합니다.')
        sys.exit(1)

    payload = {'meta': {'generated_at': NOW.strftime('%Y-%m-%d %H:%M'), 'analysis_date': d.get('meta', {}).get('analysis_date'),
                        'method': '키워드 규칙 기반 (공시 가중 1.5배, 점수 = 가중치 합 x 25, 범위 -100~100)',
                        'sources': ['OpenDART', 'Google News RSS'], 'dart_enabled': bool(corp)},
               'stocks': out}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f'[저장] {OUT} ({len(out)}종목)')
    save_snapshot(payload, d.get('meta', {}))


if __name__ == '__main__':
    main()
