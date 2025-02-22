import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from sharadar.pipeline.engine import load_sharadar_bundle, symbols, make_pipeline_engine
from zipline.pipeline.filters import StaticAssets
import time
import datetime
from sharadar.pipeline.factors import Exchange, Sector, IsDomesticCommonStock, MarketCap, Fundamentals, EV
from zipline.pipeline.factors import AverageDollarVolume

from os import environ as env
import nasdaqdatalink

import faulthandler
faulthandler.enable()

nasdaqdatalink.ApiConfig.api_key = env["NASDAQ_API_KEY"]

bundle = load_sharadar_bundle()

sectors = bundle.asset_finder.get_info([199059, 199623], 'sector')
print("sectors", sectors)

equities = bundle.asset_finder.retrieve_equities([199059, 199623])
print("equities", equities)

syms = symbols(['SPY'])
print("symbols", syms)
assert len(syms) == 1
assert syms[0].exchange == 'NYSEARCA'
assert syms[0].sid == 118691

spe = make_pipeline_engine()

pipe_start = pd.to_datetime('2020-02-05', utc=True)
pipe_end = pd.to_datetime('2020-02-07', utc=True)

pipe = Pipeline(columns={
    'Close': USEquityPricing.close.latest,
},
screen = StaticAssets(symbols(['IBM', 'F', 'AAPL']))
)

stocks = spe.run_pipeline(pipe, pipe_start, pipe_end)
print(stocks.head())
print("stocks.shape [close]", stocks.shape)
assert stocks.shape == (9, 1)

from sharadar.pipeline.factors import MarketCap, EV, Fundamentals
pipe_mkt_cap = Pipeline(columns={
    'mkt_cap': MarketCap()
},
)

start_time = time.time()
stocks = spe.run_pipeline(pipe_mkt_cap, pipe_start, pipe_end)
print(stocks.head(20))
print("stocks.shape [mkt cap]", stocks.shape)
#assert stocks.shape == (30019, 1)

pipe_mkt_cap_ev = Pipeline(columns={
    'close': USEquityPricing.close.latest,
    'sharesbas_arq': Fundamentals(field='sharesbas_arq'),
    'sharefactor_arq': Fundamentals(field='sharefactor_arq'),
    'mkt_cap': MarketCap(),
    'ev': EV(),
    'debt': Fundamentals(field='debtusd_arq'),
    'cash': Fundamentals(field='cashnequsd_arq')
},
screen = StaticAssets(symbols(['IBM', 'F', 'AAPL']))
)

stocks = spe.run_pipeline(pipe_mkt_cap_ev, pipe_start, pipe_end)
print(stocks.head())
print(stocks.shape)
#assert stocks.shape == (15, 4)

print(nasdaqdatalink.get_table('SHARADAR/SEP', date={'gte':'2020-02-03', 'lte':'2020-02-05'}, ticker='AAPL'))
print("---")
print(nasdaqdatalink.get_table('SHARADAR/SF1',
                       datekey='2020-01-29',
                       dimension='ARQ',
                       qopts={"columns":['sharefactor','sharesbas', 'price', 'marketcap']},
                       ticker='AAPL'))
print("---")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(stocks.iloc[0])
assert stocks.iloc[0]['cash']            == 39771000000
assert stocks.iloc[0]['close']           == 318.85
assert stocks.iloc[0]['debt']            == 108292000000
assert stocks.iloc[0]['ev']              == 1475010300000
assert stocks.iloc[0]['mkt_cap']         == 1406489300000
assert stocks.iloc[0]['sharefactor_arq'] == 1.00

#TODO check the factor 17501920000/4375480000=4
#assert stocks.iloc[0]['sharesbas_arq']   == 4375480000


def stocks_us():
    return (
            (USEquityPricing.close.latest > 3) &
            Exchange().element_of(['NYSE', 'NASDAQ', 'NYSEMKT']) &
            (Sector().notnull()) &
            (~Sector().element_of(['Financial Services', 'Real Estate', 'Energy', 'Utilities'])) &
            (IsDomesticCommonStock().eq(1)) &
            (Fundamentals(field='revenue_arq') > 0) &
            (Fundamentals(field='assets_arq') > 0) &
            (Fundamentals(field='equity_arq') > 0) &
            (EV() > 0)
    )

def base_universe(context):
    min_percentile = context.PARAM['min_percentile']
    return (
        (stocks_us()) &
        (AverageDollarVolume(window_length=200).percentile_between(min_percentile, 100.0, mask=stocks_us())) &
        (MarketCap().percentile_between(min_percentile, 100.0, mask=stocks_us()))
    )

pipe = Pipeline(columns={
    'Close': USEquityPricing.close.latest,
    'Exchange': Exchange(),
    'Sector': Sector()
},
screen = (
    (stocks_us()) &
    (AverageDollarVolume(window_length = 200, mask=stocks_us()).top(10))
)
)
stocks = spe.run_pipeline(pipe, pipe_end)
print(stocks.head())
print("stocks.shape", stocks.shape)
assert stocks.shape == (10, 3)


pipe_start = pd.to_datetime('2020-02-03', utc=True)
pipe_end = pd.to_datetime('2021-02-26', utc=True)
macro = symbols(['TR3M', 'TR6M', 'TR1Y', 'TR2Y', 'TR3Y', 'TR5Y', 'TR7Y', 'TR10Y','TR20Y','CBOND', 'INDPRO', 'INDPROPCT', 'PMICMP', 'UNRATE', 'RATEINF'])
pipe = Pipeline(columns={
    'Close': USEquityPricing.close.latest,
},
screen = StaticAssets(macro)
)
stocks = spe.run_pipeline(pipe, pipe_start, pipe_end)
print(stocks.head())
print("stocks.shape [close]", stocks)
assert stocks.shape == (4050, 1)