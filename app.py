import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from longport.openapi import QuoteContext, Config, Period, AdjustType

# ==========================================
# 0. 预置核心资产池
# ==========================================
PRESET_LISTS = {
    "HK": {
        "🇭🇰 恒生科技 (30)": "00700.HK, 03690.HK, 09988.HK, 01810.HK, 09618.HK, 09999.HK, 01024.HK, 02015.HK, 00981.HK, 00992.HK, 00285.HK, 06618.HK, 09626.HK, 09888.HK, 00772.HK, 03888.HK, 02382.HK, 01347.HK, 01797.HK, 02018.HK, 07800.HK, 00522.HK, 00268.HK, 00593.HK, 03032.HK, 06690.HK, 09961.HK, 02400.HK",
        "🇭🇰 恒指热门 (Top 30)": "00005.HK, 01299.HK, 00939.HK, 00941.HK, 00883.HK, 00388.HK, 02318.HK, 00016.HK, 00027.HK, 00001.HK, 01398.HK, 03988.HK, 01928.HK, 00669.HK, 01211.HK, 01088.HK, 02319.HK, 02331.HK, 01109.HK, 02628.HK, 00003.HK, 00011.HK, 00006.HK, 00857.HK, 02688.HK, 00066.HK",
        "🇭🇰 港股红利 (高股息)": "00941.HK, 00883.HK, 01088.HK, 00939.HK, 01398.HK, 03988.HK, 01288.HK, 00005.HK, 00002.HK, 00006.HK, 02628.HK, 02318.HK, 00902.HK, 00857.HK, 00386.HK"
    },
    "US": {
        "🇺🇸 纳斯达克 (100)": "INTC.US, MSFT.US, CSCO.US, KHC.US, VRTX.US, MNST.US, CDNS.US, ADSK.US, GILD.US, GOOGL.US, ADBE.US, QCOM.US, WBD.US, AMAT.US, CDNS.US, MCHP.US, ISRG.US, PAYC.US, AAPL.US, FAST.US, PCAR.US, AMZN.US, ROST.US, COST.US, LRCX.US, INTU.US, CTSH.US, KLAC.US, AMGN.US, EA.US, BIIB.US, NVDA.US, SBUX.US, AXON.US, CMCSA.US, MRVL.US, ADI.US, XOM.US, CSX.US, EXC.US, MU.US, MAR.US, AEP.US, TXN.US, CCEP.US, HON.US, AMD.US, BKR.US, PEP.US, ADP.US, KDP.US, NFLX.US, BKNG.US, ORLY.US, ROP.US, AVGO.US, NXPI.US, TSLA.US, TTWO.US, CHTR.US, COST.US, DXCM.US, FTNT.US, IDXX.US, MELI.US, PANW.US, ON.US, TMUS.US, META.US, WDAY.US, MDLZ.US, LULU.US, REGN.US, AZN.US, ASML.US, CPRT.US, ODFL.US, SNPS.US, VRSK.US, FANG.US, PANW.US, CDW.US, GOOG.US, SHOP.US, PYPL.US, TEAM.US, TTD.US, ZS.US, PDD.US, CRWD.US, DDOG.US, PLTR.US, ABNB.US, DASH.US, APP.US, GFS.US, CEG.US, GEHC.US, ARM.US, LIN.US, TRI.US",
        "🇺🇸 标普 (500)": "MMM.US, ABNB.US, ABT.US, ABBV.US, ACN.US, ADBE.US, AMD.US, AES.US, AFL.US, A.US, APD.US, AKAM.US, ALB.US, ARE.US, ALGN.US, ALLE.US, LNT.US, ALL.US, GOOGL.US, GOOG.US, MO.US, AMZN.US, AMCR.US, AEE.US, AAL.US, AEP.US, AXP.US, AIG.US, AMT.US, AWK.US, AMP.US, ABC.US, AME.US, AMGN.US, APH.US, ADI.US, ANSS.US, ANTM.US, AON.US, AOS.US, APA.US, AAPL.US, AMAT.US, APTV.US, ACGL.US, ADM.US, ANET.US, AJG.US, AIZ.US, AXON.US, T.US, ATO.US, ADSK.US, ADP.US, AZO.US, AVB.US, AVY.US, BKR.US, BLL.US, BAC.US, BBWI.US, BAX.US, BDX.US, BRK.B.US, BBY.US, BG.US, BIO.US, BIIB.US, BLDR.US, BLK.US, BK.US, BA.US, BKNG.US, BWA.US, BXP.US, BSX.US, BMY.US, AVGO.US, BR.US, BF.B.US, BX.US, CHRW.US, COG.US, CDNS.US, CZR.US, CPB.US, COF.US, CAH.US, KMX.US, CCL.US, CARR.US, CTLT.US, CAT.US, CBOE.US, CBRE.US, CDW.US, CE.US, CNC.US, CNP.US, CERN.US, CF.US, CRL.US, SCHW.US, CHTR.US, CVX.US, CMG.US, CB.US, CHD.US, CI.US, CINF.US, CTAS.US, CSCO.US, C.US, CEGVV.US, CFG.US, CLX.US, CME.US, CMS.US, KO.US, CTSH.US, CL.US, CMCSA.US, CMA.US, CAG.US, COP.US, ED.US, STZ.US, COO.US, CPRT.US, GLW.US, CTVA.US, CSGP.US, COST.US, CCI.US, CSX.US, CMI.US, CVS.US, DHI.US, DHR.US, DRI.US, DVA.US, DE.US, DAL.US, DVN.US, DXCM.US, FANG.US, DECK.US, DLR.US, DFS.US, DISCA.US, DISCK.US, DG.US, DLTR.US, D.US, DPZ.US, DOV.US, DOW.US, DTE.US, DUK.US, DD.US, EMN.US, ETN.US, EBAY.US, ECL.US, EIX.US, EW.US, EA.US, EMR.US, ENPH.US, ETR.US, EOG.US, EQT.US, EFX.US, EQIX.US, EQR.US, ESS.US, EL.US, ETSY.US, EVRG.US, ES.US, RE.US, EXC.US, EXPE.US, EXPD.US, EXR.US, XOM.US, FFIV.US, FB.US, FAST.US, FRT.US, FDX.US, FICO.US, FIS.US, FITB.US, FE.US, FISV.US, FLT.US, FMC.US, F.US, FTNT.US, FTV.US, FBHS.US, FOXA.US, FOX.US, BEN.US, FCX.US, GEHC.US, GRMN.US, IT.US, GNRC.US, GD.US, GE.US, GEV.US, GIS.US, GM.US, GPC.US, GILD.US, GL.US, GPN.US, GS.US, GWW.US, HAL.US, HBI.US, HIG.US, HAS.US, HCA.US, PEAK.US, HSIC.US, HSY.US, HES.US, HPE.US, HLT.US, HOLX.US, HD.US, HON.US, HRL.US, HST.US, HWM.US, HPQ.US, HUBB.US, HUM.US, HBAN.US, HII.US, IEX.US, IDXX.US, INFO.US, ITW.US, ILMN.US, INCY.US, IR.US, INTC.US, ICE.US, IBM.US, IP.US, IPG.US, IFF.US, INTU.US, ISRG.US, IVZ.US, INVH.US, IQV.US, IRM.US, JKHY.US, J.US, JBHT.US, JBL.US, SJM.US, JNJ.US, JCI.US, JPM.US, JNPR.US, KSU.US, K.US, KDP.US, KEY.US, KEYS.US, KMB.US, KIM.US, KMI.US, KLAC.US, KHC.US, KR.US, KVUE.US, LHX.US, LH.US, LRCX.US, LW.US, LVS.US, LEG.US, LDOS.US, LEN.US, LLY.US, LIN.US, LYV.US, LKQ.US, LMT.US, L.US, LOW.US, LULU.US, LYB.US, MTB.US, MRO.US, MPC.US, MKTX.US, MAR.US, MMC.US, MLM.US, MAS.US, MA.US, MKC.US, MCD.US, MCK.US, MDT.US, MRK.US, MET.US, MTD.US, MGM.US, MCHP.US, MU.US, MSFT.US, MAA.US, MRNA.US, MHK.US, TAP.US, MDLZ.US, MPWR.US, MNST.US, MCO.US, MS.US, MOS.US, MSI.US, MSCI.US, NDAQ.US, NTAP.US, NFLX.US, NEM.US, NWSA.US, NWS.US, NEE.US, NKE.US, NI.US, NSC.US, NTRS.US, NOC.US, NLOK.US, NCLH.US, NOV.US, NRG.US, NUE.US, NVDA.US, NVR.US, NXPI.US, ORLY.US, OXY.US, ODFL.US, OMC.US, ON.US, OKE.US, ORCL.US, OTIS.US, PCAR.US, PKG.US, PH.US, PANW.US, PAYX.US, PAYC.US, PYPL.US, PNR.US, PBCT.US, PEP.US, PKI.US, PRGO.US, PFE.US, PCG.US, PM.US, PSX.US, PNW.US, PXD.US, PNC.US, PODD.US, POOL.US, PPG.US, PPL.US, PFG.US, PG.US, PGR.US, PLD.US, PRU.US, PTC.US, PEG.US, PSA.US, PHM.US, QRVO.US, PWR.US, QCOM.US, DGX.US, RL.US, RJF.US, RTX.US, O.US, REG.US, REGN.US, RF.US, RSG.US, RMD.US, RHI.US, ROK.US, ROL.US, ROP.US, ROST.US, RCL.US, SPGI.US, CRM.US, SBAC.US, SLB.US, STLD.US, STX.US, SRE.US, NOW.US, SHW.US, SMCI.US, SPG.US, SWKS.US, SNA.US, SO.US, SOLV.US, LUV.US, SWK.US, SBUX.US, STT.US, STE.US, SYK.US, SIVB.US, SYF.US, SNPS.US, SYY.US, TECH.US, TMUS.US, TROW.US, TTWO.US, TPR.US, TRGP.US, TGT.US, TEL.US, TDY.US, TFX.US, TER.US, TSLA.US, TXN.US, TXT.US, TMO.US, TJX.US, TSCO.US, TT.US, TDG.US, TRV.US, TRMB.US, TFC.US, TYL.US, TSN.US, UBER.US, UDR.US, ULTA.US, USB.US, UNP.US, UAL.US, UNH.US, UPS.US, URI.US, UHS.US, UNM.US, VLTO.US, VLO.US, VTR.US, VRSN.US, VRSK.US, VZ.US, VRTX.US, VIAC.US, VTRS.US, V.US, VMC.US, VST.US, WRB.US, WAB.US, WMT.US, WBA.US, DIS.US, WM.US, WAT.US, WEC.US, WFC.US, WELL.US, WST.US, WDC.US, WU.US, WRK.US, WY.US, WMB.US, WLTW.US, WYNN.US, XEL.US, XLNX.US, XYL.US, YUM.US, ZBRA.US, ZBH.US, ZTS.US",
    }
}


# ==========================================
# 1. 配置与辅助函数
# ==========================================
SETTINGS_FILE = "stock_settings.json"

def load_settings():
    default = {
        "HK": PRESET_LISTS["HK"]["🇭🇰 恒生科技 (30)"],
        "US": PRESET_LISTS["US"]["🇺🇸 纳斯达克 (100)"],
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding='utf-8') as f:
                saved = json.load(f)
                for k, v in default.items():
                    if k not in saved: saved[k] = v
                return saved
        except:
            return default
    return default

def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

def format_large_num(num):
    """亿/万 格式化"""
    if num is None or num == 0: return "-"
    abs_num = abs(num)
    if abs_num >= 100000000: return f"{num / 100000000.0:.2f}亿"
    elif abs_num >= 10000: return f"{num / 10000.0:.2f}万"
    else: return f"{num:.2f}"

# ==========================================
# 2. 核心算法
# ==========================================

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_atr(df, length):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean() 

def knn_one_step(target_val, history_st_values, history_labels, k):
    n = len(history_st_values)
    distances = []
    for i in range(n):
        dist = abs(target_val - history_st_values[i])
        distances.append((dist, history_labels[i]))
    distances.sort(key=lambda x: x[0]) 
    
    weighted_sum = 0.0
    total_weight = 0.0
    for i in range(min(k, n)):
        dist, label = distances[i]
        weight = 1.0 / (dist + 1e-6) 
        weighted_sum += weight * label
        total_weight += weight
    
    if total_weight == 0: return 0
    return weighted_sum / total_weight

def calculate_trend_status(df, target_direction):
    """
    筛选逻辑: 
    1. 价格 vs 周EMA20
    2. 日EMA20 vs 日EMA60
    3. SuperTrend AI 信号
    """
    # 1. 周线 EMA20
    df['time'] = pd.to_datetime(df['time'])
    df_weekly = df.set_index('time').resample('W').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    df_weekly.dropna(subset=['close'], inplace=True)
    df_weekly['ema20'] = df_weekly['close'].ewm(span=20, adjust=False).mean()
    
    if len(df_weekly) < 20: return False, 0, 0
        
    w_ema_val = df_weekly['ema20'].iloc[-1]
    current_price = df['close'].iloc[-1]
    
    # 2. 日线 EMA20 vs EMA60
    ema20_daily = calculate_ema(df['close'], 20)
    ema60_daily = calculate_ema(df['close'], 60)
    d_ema20_val = ema20_daily.iloc[-1]
    d_ema60_val = ema60_daily.iloc[-1]
    
    # 过滤条件
    if target_direction == 'Bullish':
        if current_price <= w_ema_val: return False, 0, w_ema_val
        if d_ema20_val <= d_ema60_val: return False, 0, w_ema_val
    else:
        if current_price >= w_ema_val: return False, 0, w_ema_val
        if d_ema20_val >= d_ema60_val: return False, 0, w_ema_val

    # 3. SuperTrend AI
    st_len = 10
    st_factor = 3.0
    k = 3
    n_param = 10
    n = max(k, n_param) 
    knn_price_len = 20
    knn_st_len = 100
    
    if len(df) < max(knn_st_len, 200): return False, 0, 0
    
    cv = df['close'] * df['volume']
    ema_cv = calculate_ema(cv, st_len)
    ema_vol = calculate_ema(df['volume'], st_len)
    vwma = ema_cv / ema_vol.replace(0, np.nan)
    
    atr = calculate_atr(df, st_len)
    basic_upper = vwma + st_factor * atr
    basic_lower = vwma - st_factor * atr
    
    close_arr = df['close'].values
    bu_arr = basic_upper.values
    bl_arr = basic_lower.values
    st_lower = np.full(len(df), np.nan)
    st_upper = np.full(len(df), np.nan)
    supertrend = np.zeros(len(df))
    trends = np.zeros(len(df)) 
    
    final_lower = bl_arr[0] if not np.isnan(bl_arr[0]) else 0
    final_upper = bu_arr[0] if not np.isnan(bu_arr[0]) else 0
    
    for i in range(1, len(df)):
        prev_lb = st_lower[i-1] if not np.isnan(st_lower[i-1]) else bl_arr[i]
        if not np.isnan(bl_arr[i]):
            final_lower = bl_arr[i] if (bl_arr[i] > prev_lb or close_arr[i-1] < prev_lb) else prev_lb
        st_lower[i] = final_lower

        prev_ub = st_upper[i-1] if not np.isnan(st_upper[i-1]) else bu_arr[i]
        if not np.isnan(bu_arr[i]):
            final_upper = bu_arr[i] if (bu_arr[i] < prev_ub or close_arr[i-1] > prev_ub) else prev_ub
        st_upper[i] = final_upper
        
        prev_trend = trends[i-1]
        if prev_trend == -1: 
            trend = 1 if close_arr[i] > final_upper else -1
        else: 
            trend = -1 if close_arr[i] < final_lower else 1
        trends[i] = trend
        supertrend[i] = final_lower if trend == 1 else final_upper

    df['supertrend'] = supertrend
    df['knn_price'] = calculate_ema(df['close'], knn_price_len)
    df['knn_st'] = calculate_ema(df['supertrend'], knn_st_len)
    df['raw_label'] = np.where(df['knn_price'] > df['knn_st'], 1, 0)
    
    def get_ai_label(idx):
        if idx < n + 50: return -1
        slice_st = df['supertrend'].iloc[idx-(n-1) : idx+1].values
        slice_labels = df['raw_label'].iloc[idx-(n-1) : idx+1].values
        current_val = df['supertrend'].iloc[idx]
        ai_score = knn_one_step(current_val, slice_st, slice_labels, k)
        
        if ai_score > 0.99: return 1
        if ai_score < 0.01: return 0
        return -1 
    
    target_val = 1 if target_direction == 'Bullish' else 0
    current_idx = len(df) - 1
    
    if get_ai_label(current_idx) != target_val:
        return False, 0, w_ema_val
    
    duration = 0
    # 移除100天限制，扫描所有可用数据
    scan_limit = len(df) - n - 50 
    for i in range(scan_limit):
        if get_ai_label(current_idx - i) == target_val:
            duration += 1
        else:
            break
            
    return True, duration, w_ema_val

# ==========================================
# 3. 页面与交互逻辑
# ==========================================

st.set_page_config(page_title="SuperTrend AI Pro", page_icon="⚡", layout="wide")
settings = load_settings()

@st.cache_resource
def get_config():
    return Config.from_env()

st.sidebar.header("⚙️ 策略与股票池")

st.sidebar.subheader("1. 趋势方向")
st.sidebar.info("逻辑: 价格>20周EMA 且 日EMA20>60 且 SuperTrend看涨")
direction_option = st.sidebar.radio("寻找机会:", ("🚀 看涨 (Bullish)", "📉 看跌 (Bearish)"), index=0)
target_trend = "Bullish" if "看涨" in direction_option else "Bearish"

st.sidebar.divider()
st.sidebar.subheader("2. 市场筛选 (自动填充)")

tab_hk, tab_us = st.sidebar.tabs(["🇭🇰 港股", "🇺🇸 美股"])

trigger_market = None
symbol_list_to_run = []
market_display_name = ""

def handle_preset_selection(market_key, initial_val, widget_key):
    options = ["📝 自定义/手动输入"] + list(PRESET_LISTS[market_key].keys())
    sel_key = f"sel_{widget_key}"
    txt_key = f"input_{widget_key}"
    
    default_idx = 0
    for i, opt in enumerate(options):
        if opt in PRESET_LISTS[market_key] and PRESET_LISTS[market_key][opt] == initial_val:
            default_idx = i
            break
    
    def on_selection_change():
        selection = st.session_state[sel_key]
        if selection != "📝 自定义/手动输入":
            st.session_state[txt_key] = PRESET_LISTS[market_key][selection]

    st.selectbox(f"快速选择 {market_key}:", options, index=default_idx, key=sel_key, on_change=on_selection_change)
    if txt_key not in st.session_state: st.session_state[txt_key] = initial_val
    return st.text_area("代码列表:", key=txt_key, height=120)

with tab_hk:
    hk_codes = handle_preset_selection("HK", settings.get("HK", ""), "hk")
    if st.button("🚀 筛选港股", type="primary", use_container_width=True):
        trigger_market = "HK"
        symbol_list_to_run = [s.strip() for s in hk_codes.replace('\n', ',').split(',') if s.strip()]
        market_display_name = "港股 (HK)"

with tab_us:
    us_codes = handle_preset_selection("US", settings.get("US", ""), "us")
    if st.button("🚀 筛选美股", type="primary", use_container_width=True):
        trigger_market = "US"
        symbol_list_to_run = [s.strip() for s in us_codes.replace('\n', ',').split(',') if s.strip()]
        market_display_name = "美股 (US)"

# --- 主页面 ---
trend_icon = "🟢" if target_trend == "Bullish" else "🔴"
st.title(f"{trend_icon} SuperTrend AI Pro - {target_trend}")

if 'screened_results' not in st.session_state:
    st.session_state.screened_results = []
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'current_market_scope' not in st.session_state:
    st.session_state.current_market_scope = ""

if trigger_market:
    new_settings = {"HK": hk_codes, "US": us_codes}
    save_settings(new_settings)
    st.session_state.current_market_scope = market_display_name
    
    msg_placeholder = st.empty()
    bar_placeholder = st.empty()
    
    msg_placeholder.info(f"⏳ 正在扫描 **{market_display_name}** ({len(symbol_list_to_run)} 只股票)...")
    progress_bar = bar_placeholder.progress(0)
    
    ctx = QuoteContext(get_config())
    results = []
    total = len(symbol_list_to_run)
    
    if total == 0:
        msg_placeholder.error("❌ 股票列表为空。")
        bar_placeholder.empty()
    else:
        for idx, symbol in enumerate(symbol_list_to_run):
            progress_bar.progress((idx + 1) / total)
            try:
                candles = ctx.candlesticks(symbol, Period.Day, 500, AdjustType.ForwardAdjust)
                if len(candles) > 200:
                    df = pd.DataFrame([{
                        "time": c.timestamp, "open": float(c.open), "high": float(c.high), 
                        "low": float(c.low), "close": float(c.close), "volume": float(c.volume)
                    } for c in candles])
                    
                    is_match, duration, w_ema = calculate_trend_status(df, target_trend)
                    
                    if is_match:
                        results.append({
                            "symbol": symbol,
                            "trend": target_trend,
                            "duration": duration,
                            "last_close_algo": df['close'].iloc[-1],
                            "weekly_ema": w_ema
                        })
            except Exception as e:
                pass 
        
        st.session_state.screened_results = results
        st.session_state.last_update_time = time.time()
        
        msg_placeholder.empty()
        bar_placeholder.empty()
        
        if not results:
            st.warning(f"在 {market_display_name} 中未发现符合条件的股票。")
        else:
            st.success(f"✅ 筛选完成！找到 {len(results)} 只优质机会。")

# --- 结果展示 (表格优化) ---
if st.session_state.screened_results:
    if st.session_state.current_market_scope:
        st.subheader(f"📊 筛选结果: {st.session_state.current_market_scope}")
    
    target_symbols = [item['symbol'] for item in st.session_state.screened_results]
    
    if target_symbols:
        ctx = QuoteContext(get_config())
        try:
            quotes = ctx.quote(target_symbols)
            static_infos = ctx.static_info(target_symbols)
            
            quote_map = {q.symbol: q for q in quotes}
            info_map = {i.symbol: i for i in static_infos}
            
            display_data = []
            
            for item in st.session_state.screened_results:
                sym = item['symbol']
                q = quote_map.get(sym)
                info = info_map.get(sym)
                
                current = float(q.last_done) if q else item['last_close_algo']
                prev = float(q.prev_close) if q else item['last_close_algo']
                chg = ((current - prev) / prev) * 100 if prev > 0 else 0.0
                
                total_shares = int(info.total_shares) if info and info.total_shares else 0
                eps_ttm = 0.0
                if info and info.eps_ttm:
                    try: eps_ttm = float(info.eps_ttm)
                    except: eps_ttm = 0.0
                
                mkt_cap = current * total_shares
                pe_ttm = current / eps_ttm if eps_ttm > 0 else -1
                
                # 计算乖离率
                w_ema = item['weekly_ema']
                bias = ((current - w_ema) / w_ema) * 100 if w_ema > 0 else 0
                
                icon = "🟢" if item['trend'] == "Bullish" else "🔴"
                display_data.append({
                    "代码": sym, 
                    "最新价": current, 
                    "涨跌幅 (%)": chg,
                    "偏离度(周EMA)": bias, # 保持数值以便格式化
                    "总市值": format_large_num(mkt_cap),
                    "市盈率(TTM)": f"{pe_ttm:.2f}" if pe_ttm > 0 else "亏损",
                    "趋势": f"{icon} {item['trend']}", 
                    "持续天数": int(item['duration'])
                })
            
            df_display = pd.DataFrame(display_data)
            
            def color_change(val):
                return f'color: {"#00CC00" if val >= 0 else "#FF3333"}; font-weight: bold;'
            def color_trend_col(val):
                return f'color: {"#00AA00" if "Bullish" in val else "#CC0000"}; font-weight: bold;'

            st_df = df_display.style.format({
                "最新价": "{:.3f}", 
                "涨跌幅 (%)": "{:+.2f}%", 
                "偏离度(周EMA)": "{:+.2f}%",
                "持续天数": "{} 天"
            }).map(color_change, subset=["涨跌幅 (%)"]).map(color_trend_col, subset=["趋势"])

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🔄 刷新行情"): st.rerun()
            with col2:
                if st.session_state.last_update_time:
                    st.caption(f"上次筛选: {time.ctime(st.session_state.last_update_time)}")
            
            st.dataframe(st_df, use_container_width=True, height=600)

        except Exception as e:
            st.error(f"行情数据获取失败: {e}")

elif not trigger_market and not st.session_state.screened_results:
    st.info("👈 请在左侧选择【预设股票池】并点击筛选。")
