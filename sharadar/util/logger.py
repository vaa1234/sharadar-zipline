import datetime
import os
import subprocess
import sys
from os import environ as env

from logbook import Logger, FileHandler, DEBUG, INFO, NOTSET, StreamHandler, set_datetime_format
from sharadar.util.mail import send_mail
from zipline.api import get_datetime
import pandas as pd

def now_time():
    return pd.to_datetime("today")

# log in local time instead of UTC
set_datetime_format("local")
LOG_ENTRY_FMT = '[{record.time:%Y-%m-%d %H:%M:%S}] {record.level_name}: {record.message}'
LOG_LEVEL_MAP = {'CRITICAL': 2, 'ERROR': 3, 'WARNING': 4, 'NOTICE': 5, 'INFO': 6, 'DEBUG': 7, 'TRACE': 7}

class SharadarDbBundleLogger(Logger):
    def __init__(self, logname='sharadar_db_bundle', level=NOTSET):
        super().__init__(logname, level)

        now = datetime.datetime.now()
        self.filename = os.path.join(env["HOME"], "log",
                                   "sharadar-zipline" + '_' + now.strftime('%Y-%m-%d_%H%M') + ".log")

        log_file_handler = FileHandler(self.filename, level=DEBUG, bubble=True)
        log_file_handler.format_string = LOG_ENTRY_FMT
        self.handlers.append(log_file_handler)

        log_std_handler = StreamHandler(sys.stdout, level=INFO)
        log_std_handler.format_string = LOG_ENTRY_FMT
        self.handlers.append(log_std_handler)

    def process_record(self, record):
        super().process_record(record)

class StrategyLogger(Logger):
    def __init__(self, filename, arena='backtest', logname='Backtest', level=NOTSET, record_time=get_datetime):
        super().__init__(logname, level)

        path, ext = os.path.splitext(filename)
        now = datetime.datetime.now()
        log_filename = path + '_' + now.strftime('%Y-%m-%d_%H%M') + ".log"
        file_handler = FileHandler(log_filename, level=DEBUG, bubble=True)
        file_handler.format_string = LOG_ENTRY_FMT
        self.handlers.append(file_handler)

        stream_handler = StreamHandler(sys.stdout, level=INFO)
        stream_handler.format_string = LOG_ENTRY_FMT
        self.handlers.append(stream_handler)

        self.arena = arena

        if self.arena == "backtest":
            self.record_time = get_datetime
        else:
            self.record_time = now_time

    def process_record(self, record):
        """
        use the date of the trading day for log purposes
        """
        super().process_record(record)
        record.time = self.record_time()

log = SharadarDbBundleLogger()

if __name__ == '__main__':
    log = StrategyLogger()
    log.info("Hello World!")
    log.error("ciao")
    log.warning("ciao\nbello")

    StrategyLogger(__file__, arena='live', logname="Myname", record_time=now_time).warn("Hello World!")

