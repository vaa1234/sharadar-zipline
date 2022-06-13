import pandas as pd
from exchange_calendars import get_calendar


def last_trading_date(date:str = pd.to_datetime("today").strftime('%Y-%m-%d'), calendar = get_calendar('XNYS')):
    """
    The last trading date as of the given date (default: today)
    """
    dt = pd.to_datetime(date)
    return date if calendar.is_session(dt) else calendar.previous_open(dt).strftime('%Y-%m-%d')

def isopen_after_time(calendar_name, offset):

    cal = get_calendar(calendar_name)

    future_time = (pd.Timestamp.utcnow() + pd.Timedelta(offset)).strftime('%Y-%m-%d %H:%M')

    return cal.is_trading_minute(future_time)