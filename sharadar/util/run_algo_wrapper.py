import pandas as pd
import os
from os import environ as env
import importlib.machinery
from sharadar.util.run_algo import run_algorithm
from sharadar.live.brokers.ib_broker import IBBroker
import pickle
import psutil
import time
import multiprocessing as mp
from multiprocessing import Process
import threading
from pathlib import Path
from sharadar.util.logger import SharadarLogger
log = SharadarLogger(Path(__file__).stem)

STRATEGIES_PATH = os.path.join(env["HOME"], "src/python/strategies/zipline/") # directory with zipline strategies
STRATEGIES_DATABASE = os.path.join(env["HOME"], '.zipline', 'strategies_db.pickle') # pickle file for track running strategies

class LiveTradeManager:

    def __init__(self, strategy_name, tws_uri=None, bundle='sharadar', freq='minute'):

        if not os.path.isfile(STRATEGIES_DATABASE):
            self.create_database()

        self.strategies_db = self.read_database()
        self.strategy_name = strategy_name
        self.tws_uri = tws_uri
        self.bundle = bundle
        self.freq = freq

    def create_database(self):

        data = {'launched': {}}

        with open(STRATEGIES_DATABASE, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_database(self):
        with open(STRATEGIES_DATABASE, 'wb') as handle:
            pickle.dump({'launched': self.strategies_db}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read_database(self):
        return pd.read_pickle(STRATEGIES_DATABASE)['launched']

    def start(self):

        if self.is_alive():
            err_msg = f'Strategy {self.strategy_name} is already running'
            log.error(err_msg)
            raise Exception(err_msg)

        if self.tws_uri is None:
            err_msg = 'tws_uri must be set for live trading. Example: localhost:7496:1234'
            log.error(err_msg)
            raise Exception(err_msg)

        strategy_file = STRATEGIES_PATH + self.strategy_name + '.py'
        if not os.path.isfile(strategy_file):
            err_msg = f'{strategy_file} not exist'
            log.error(err_msg)
            raise OSError(err_msg)

        strategy = importlib.machinery.SourceFileLoader('strategy', strategy_file).load_module()

        log.info(f'Trade strategy {self.strategy_name}, bundle: {self.bundle}, twsuri: {self.tws_uri}, freq: {self.freq}')

        run_algorithm_kwargs = {
            'handle_data': strategy.handle_data,
            'analyze': strategy.analyze,
            'initialize': strategy.initialize,
            'before_trading_start': strategy.before_trading_start,
            'data_frequency': self.freq,
            'broker': IBBroker(tws_uri=self.tws_uri),
            'state_filename': os.path.join(STRATEGIES_PATH, self.strategy_name + '_statefile'),
            'bundle': self.bundle,
        }
        
        thread = threading.Thread(target=run_algorithm, kwargs=run_algorithm_kwargs)
        thread.daemon = True
        thread.start()

        # add pid to db
        self.strategies_db[self.strategy_name] = os.getpid()

        time.sleep(0.5)

        if self.is_alive():
            log.info(f'Strategy {self.strategy_name} started successfully')

        self.save_database()

    def stop(self):
        if self.strategies_db.get(self.strategy_name) is None:
            err_msg = f'Strategy {self.strategy_name} not running'
            log.error(err_msg)
            raise Exception(err_msg)

        pid = self.strategies_db[self.strategy_name]
        
        if psutil.pid_exists(pid):

            p = psutil.Process(pid)
            p.terminate()

            time.sleep(0.3)

            if not self.is_alive():

                del self.strategies_db[self.strategy_name] # remove pid from db
                self.save_database()

                log.info(f'Strategy {self.strategy_name} stoped')

                return True
            else:
                return False

    def is_alive(self):
        if self.strategies_db.get(self.strategy_name) is not None and psutil.pid_exists(self.strategies_db[self.strategy_name]):
            return True
        else:
            return False

def backtest(strategy_name, start, end, bundle='sharadar', start_capital=100000, freq='daily'):

    strategy_file = STRATEGIES_PATH + strategy_name + '.py'
    if not os.path.isfile(strategy_file):
        err_msg = f'{strategy_file} not exist'
        log.error(err_msg)
        raise OSError(err_msg)
    
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

    strategy = LiveTradeManager(strategy_name, tws_uri, bundle, freq)
    strategy.start()

    if strategy.is_alive():
        return True
    else:
        return False

def list_active_strategies():

    try:
        strategies_db = pd.read_pickle(STRATEGIES_DATABASE)['launched']
    except:
        raise
    
    active_strategies = []

    for strategy_name, pid in strategies_db.items():

        if psutil.pid_exists(pid):
            active_strategies.append(strategy_name)

    return active_strategies

def statefile_exist(strategy_name):
    return True if os.path.isfile(os.path.join(STRATEGIES_PATH, strategy_name + '_statefile')) else False