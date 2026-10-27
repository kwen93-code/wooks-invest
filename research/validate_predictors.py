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
FEATURES=["mom_12","mom_6","mom_3","vol_12","max_drawdown_12","ten_year","yield_slope","vix"]
FRED={"ten_year":"DGS10","two_year":"DGS2","vix":"VIXCLS"}

def get_fred(series):
    url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    x=pd.read_csv(url)
    x.columns=["date",series]
    x["date"]=pd.to_datetime(x["date"])
    x[series]=pd.to_numeric(x[series].replace(".",np.nan),errors="coerce")
    return x.set_index("date")[series].resample("ME").last().ffill()

def get_prices():
    raw=yf.download(TICKERS,start="1999-01-01",auto_adjust=True,actions=False,
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
    # Forecast at Dec month-end only; training uses targets that had already matured.
    dates=[t for t in df.index if t.month==12 and t.year>=2017 and pd.notna(df.loc[t,"target"])]
    out=[]
    for t in dates:
        cutoff=t-pd.DateOffset(months=HORIZON)
        train=df[(df.index<=cutoff)&df.target.notna()].copy()
        if len(train)<72: continue
        test=df.loc[[t]]
        baseline=float(train["target"].mean())
        pred=fit_predict(train,test,FEATURES)
        y=float(test.target.iloc[0])
        row={"forecast_date":str(t.date()),"actual":y,"baseline":baseline,"ridge":pred}
        for feature in FEATURES:
            cols=[x for x in FEATURES if x!=feature]
            row["omit_"+feature]=fit_predict(train,test,cols)
        out.append(row)
    if len(out)<4: return None
    d=pd.DataFrame(out)
    mae=lambda key:round(float((d[key]-d.actual).abs().mean())*100,3)
    base=mae("baseline"); ridge=mae("ridge")
    ablation=[{"feature":f,"mae_without_pct_points":mae("omit_"+f),
               "removal_mae_delta_pct_points":round(mae("omit_"+f)-ridge,3)}
              for f in FEATURES]
    ablation.sort(key=lambda x:x["removal_mae_delta_pct_points"],reverse=True)
    return {"test_count":len(out),"test_start":out[0]["forecast_date"],
            "test_end":out[-1]["forecast_date"],"baseline_mae_pct_points":base,
            "ridge_mae_pct_points":ridge,
            "improvement_pct_points":round(base-ridge,3),
            "model_wins":ridge<base,"feature_ablation":ablation,
            "annual_test_predictions":out}

def main():
    close=get_prices()
    macros={k:get_fred(v) for k,v in FRED.items()}
    macro=pd.DataFrame(macros)
    macro["yield_slope"]=macro["ten_year"]-macro["two_year"]
    result={"generated_at":datetime.now(timezone.utc).isoformat(),
            "method":"Annual December walk-forward, next 12 month adjusted-close total return; expanding Ridge(alpha=20) vs historical mean",
            "source":"Yahoo Finance adjusted closes; FRED DGS10 DGS2 VIXCLS",
            "limitations":["Total returns in USD before Korean withholding/capital gains tax and FX.",
                           "Yahoo adjusted-close series and latest revised macro series are not strictly point-in-time vintage.",
                           "Variable omission is conditional predictive contribution, not causal effect.",
                           "Only 1-year ahead forecasts are validated; longer-horizon claims require separate tests."],
            "etfs":{}}
    for ticker in TICKERS:
        df=frame_for(ticker,close,macro)
        if df is None: result["etfs"][ticker]={"error":"Insufficient data"};continue
        result["etfs"][ticker]=evaluate(df) or {"error":"Insufficient completed out-of-sample forecasts"}
    Path("research").mkdir(exist_ok=True)
    Path("research/validation_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for symbol,r in result["etfs"].items():
        print(symbol,r.get("baseline_mae_pct_points"),r.get("ridge_mae_pct_points"),r.get("improvement_pct_points"))
    if not any("ridge_mae_pct_points" in r for r in result["etfs"].values()):raise RuntimeError("No ETF passed data/validation checks")
if __name__=="__main__":main()
