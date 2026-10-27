#!/usr/bin/env python3
"""Walk-forward annualized 12-month ETF return forecasts; no look-ahead."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TICKERS=["QQQ","VOO","SCHD","QLD"]
HORIZON=12
FEATURES=["mom_12","mom_6","mom_3","vol_12","max_drawdown_12","ten_year","yield_slope","vix","usd_index_mom_12","global_equity_mom_12","emerging_equity_mom_12","oil_mom_12","gold_mom_12","global_us_relative_12"]
FRED={"ten_year":"DGS10","two_year":"DGS2","vix":"VIXCLS"}
GLOBAL_TICKERS={"usd_index":"UUP","global_equity":"VEU","emerging_equity":"EEM","oil":"USO","gold":"GLD"}

def get_fred(series):
    url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    x=pd.read_csv(url)
    x.columns=["date",series]
    x["date"]=pd.to_datetime(x["date"])
    x[series]=pd.to_numeric(x[series].replace(".",np.nan),errors="coerce")
    return x.set_index("date")[series].resample("ME").last().ffill()

def get_prices():
    raw=yf.download(TICKERS+list(GLOBAL_TICKERS.values()),start="1999-01-01",auto_adjust=True,actions=False,
                    progress=False,threads=False)
    if raw.empty: raise RuntimeError("ETF prices unavailable; no fabricated data")
    close=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":TICKERS[0]})
    return close.resample("ME").last()

def frame_for(ticker,close,macro):
    px=close[ticker].dropna()
    if len(px)<130: return None
    r=px.pct_change()
    f=pd.DataFrame(index=px.index)
    for n in (3,6,12): f[f"mom_{n}"]=px.pct_change(n)
    f["vol_12"]=r.rolling(12).std()*np.sqrt(12)
    f["max_drawdown_12"]=px.div(px.rolling(12).max()).sub(1).rolling(12).min()
    f=f.join(macro.reindex(f.index,method="ffill"))
    f["target"]=px.shift(-HORIZON).div(px).sub(1)
    return f.dropna(subset=FEATURES)

def fit_predict(train,obs,cols):
    m=make_pipeline(StandardScaler(),Ridge(alpha=20.0))
    m.fit(train[cols],train["target"])
    return float(m.predict(obs[cols])[0])

def evaluate(df):
    # Use non-overlapping annual December forecasts, with only matured labels.
    dates=[t for t in df.index if t.month==12 and t.year>=2017 and pd.notna(df.loc[t,"target"])]
    out=[]
    candidates={"mean":[],"ridge_all":FEATURES,
                **{f"single_{f}":[f] for f in FEATURES},
                "mom_6_vol":["mom_6","vol_12"],
                "mom_12_vix":["mom_12","vix"],
                "vol_vix":["vol_12","vix"]}
    for t in dates:
        cutoff=t-pd.DateOffset(months=HORIZON)
        train=df[(df.index<=cutoff)&df.target.notna()].copy()
        if len(train)<72:continue
        test=df.loc[[t]]
        row={"forecast_date":str(t.date()),"actual":float(test.target.iloc[0])}
        for name,cols in candidates.items():
            row[name]=fit_predict(train,test,cols) if cols else float(train.target.mean())
        out.append(row)
    if len(out)<4:return None
    d=pd.DataFrame(out)
    scores={name:round(float((d[name]-d.actual).abs().mean())*100,3) for name in candidates}
    ranking=sorted(scores,key=scores.get)
    baseline=scores["mean"]
    return {"test_count":len(out),"test_start":out[0]["forecast_date"],
            "test_end":out[-1]["forecast_date"],
            "baseline_mae_pct_points":baseline,
            "candidate_mae_pct_points":scores,
            "ranking_by_mae":ranking,
            "best_in_sample_selection_warning":"Ranking is descriptive only. Selecting the winner on these test years and reporting its MAE as unbiased performance is invalid; nested walk-forward selection is needed.",
            "best_candidate":ranking[0],
            "best_candidate_improvement_pct_points":round(baseline-scores[ranking[0]],3),
            "annual_test_predictions":out}

def main():
    close=get_prices()
    macros={k:get_fred(v) for k,v in FRED.items()}
    macro=pd.DataFrame(macros)
    macro["yield_slope"]=macro["ten_year"]-macro["two_year"]
    for name,symbol in GLOBAL_TICKERS.items():
        macro[name+"_mom_12"]=close[symbol].pct_change(12)
    macro["global_us_relative_12"]=macro["global_equity_mom_12"]-close["VOO"].pct_change(12)
    result={"generated_at":datetime.now(timezone.utc).isoformat(),
            "method":"Annual December walk-forward, 12-month adjusted-close total returns; historical mean versus prespecified single-feature and small Ridge models",
            "source":"Yahoo Finance ETF adjusted closes (UUP, VEU, EEM, USO, GLD international proxies); FRED DGS10 DGS2 VIXCLS",
            "limitations":["Total returns in USD before Korean withholding/capital gains tax and FX.",
                           "Yahoo adjusted-close series and latest revised macro series are not strictly point-in-time vintage.",
                           "Variable omission is conditional predictive contribution, not causal effect.",
                           "Only 1-year ahead forecasts are validated; longer-horizon claims require separate tests.",
                           "International variables use investable ETF proxies rather than spot FX, crude futures or full global indices.",
                           "USO futures-roll effects and UUP currency basket composition limit macroeconomic interpretation."],
            "etfs":{}}
    for ticker in TICKERS:
        df=frame_for(ticker,close,macro)
        if df is None: result["etfs"][ticker]={"error":"Insufficient data"};continue
        result["etfs"][ticker]=evaluate(df) or {"error":"Insufficient completed out-of-sample forecasts"}
    Path("research").mkdir(exist_ok=True)
    Path("research/validation_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for symbol,r in result["etfs"].items():
        print(symbol,r.get("baseline_mae_pct_points"),r.get("best_candidate"),r.get("best_candidate_improvement_pct_points"))
    if not any("candidate_mae_pct_points" in r for r in result["etfs"].values()):raise RuntimeError("No ETF passed data/validation checks")
if __name__=="__main__":main()
