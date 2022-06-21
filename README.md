Sqlite based zipline bundle for the Sharadar datasets SEP, SFP and SF1.

Unlike the standard zipline bundles, it allows incremental updates, because sql tables are used instead of bcolz.

Step 1. Make sure you can access Quandl, and you have a Quandl api key. I have set my Quandl api key as an environment variable.

>export NASDAQ_API_KEY="your API key"  

Step 2. Clone or download the code and install it using:

>python setup.py install 

For zipline in order to build the cython files run:
>python setup.py build_ext --inplace

Add this code to your ~/.zipline/extension.py:
```python
from zipline.data import bundles
from zipline.finance import metrics
from sharadar.loaders.ingest_sharadar import from_nasdaqdatalink
from sharadar.util.metric_daily import default_daily

bundles.register("sharadar", from_nasdaqdatalink(), create_writers=False)
metrics.register('default_daily', default_daily)
```

The new entry point is **sharadar-zipline** (it replaces *zipline*).

To ingest price and fundamental data every day at 9:00 using cron
> 0 9 * * * sharadar-zipline bundle -i sharadar

To backtest strategy called 'test'
> sharadar-zipline backtest -n test -s 2020-01-01 -e 2022-03-01 -b sharadar -f daily -c 100000

To run an algorithm in live trading mode if market open in 30 minutes
> 5 16 * * mon-fri sharadar-zipline calendar XNYS isopen --after 30m && sharadar-zipline trade -n test --twsuri 127.0.0.1:7496:143


Example using pipeline with fundamentals factors:
```python
from zipline.pipeline import Pipeline
import pandas as pd
from sharadar.pipeline.factors import (
    MarketCap,
    EV,
    Fundamentals
)
from sharadar.pipeline.engine import symbol, symbols, make_pipeline_engine
from zipline.pipeline.filters import StaticAssets

pipe = Pipeline(columns={
    'mkt_cap': MarketCap(),
    'ev': EV(),
    'debt': Fundamentals(field='debtusd_arq'),
    'cash': Fundamentals(field='cashnequsd_arq')
},
screen = StaticAssets(symbols(['IBM', 'F', 'AAPL']))
)
spe = make_pipeline_engine()

pipe_date = pd.to_datetime('2020-02-03', utc=True)

stocks = spe.run_pipeline(pipe, pipe_date)
stocks
```
