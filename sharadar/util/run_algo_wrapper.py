import pandas as pd
import sys
import os
from os import environ as env
import importlib.machinery
from sharadar.util.run_algo import run_algorithm
from sharadar.live.brokers.ib_broker import IBBroker

STRATEGIES_PATH = os.path.join(env["HOME"], "src/python/strategies/zipline/")

def backtest(strategy_name, start, end, bundle='sharadar', start_capital=100000, freq='daily'):

    strategy_file = STRATEGIES_PATH + strategy_name + '.py'
    if not os.path.isfile(strategy_file):
        raise OSError(f'{strategy_file} not exist')    

    strategy = importlib.machinery.SourceFileLoader('strategy', strategy_file).load_module()

    run_algorithm(
        start = pd.to_datetime(start, utc=True),
        end = pd.to_datetime(end, utc=True),
        handle_data = strategy.handle_data,
        analyze = strategy.analyze,
        initialize = strategy.initialize,
        before_trading_start = strategy.before_trading_start,
        capital_base = start_capital,
        bundle=bundle,
        data_frequency = freq,
        benchmark_symbol = None,
    )

def trade(strategy_name, tws_uri=None, bundle='sharadar', freq='minute'):

    if tws_uri is None:
        raise Exception('tws_uri must be set for live trading. Example: localhost:7496:1234')

    strategy_file = STRATEGIES_PATH + strategy_name + '.py'
    if not os.path.isfile(strategy_file):
        raise OSError(f'{strategy_file} not exist')    

    strategy = importlib.machinery.SourceFileLoader('strategy', strategy_file).load_module()

    run_algorithm(
        handle_data = strategy.handle_data,
        analyze = strategy.analyze,
        initialize = strategy.initialize,
        before_trading_start = strategy.before_trading_start,
        data_frequency = freq,
        broker = IBBroker(tws_uri=tws_uri),
        state_filename = os.path.join(STRATEGIES_PATH, strategy_name + '_statefile'),
        bundle = bundle,
    )