from datetime import datetime
import argparse
import os
import sys
from zipline.data.bundles import ingest
from sharadar.util.run_algo import load_extensions
from sharadar.util.run_algo_wrapper import backtest as run_backtest, trade as run_trade
from sharadar.util.calendar_util import isopen_after_time

def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)

def main(command_line=None):
    parser = argparse.ArgumentParser()
    
    subparser = parser.add_subparsers(dest='command')
    backtest = subparser.add_parser('backtest')
    trade = subparser.add_parser('trade')
    bundle = subparser.add_parser('bundle')
    calendar = subparser.add_parser('calendar')
    
    calendar.add_argument('name')
    calendar.add_argument('state', choices=['isopen'])
    calendar.add_argument('-a', '--after', type=str, metavar='timedelta', required=True)

    backtest.add_argument('-n', '--strategyname', type=str, required=True)
    backtest.add_argument('-s', '--start', type=valid_date, required=True)
    backtest.add_argument('-e', '--end', type=valid_date, required=True)
    backtest.add_argument('-b', '--bundle', type=str, required=False, default='sharadar')
    backtest.add_argument('-c', '--capital', type=int, required=False, default=100000)
    backtest.add_argument('-f', '--freq', type=str, required=False, default='daily')

    trade.add_argument('-n', '--strategyname', type=str, required=True)
    trade.add_argument('--twsuri', type=str, required=True)
    trade.add_argument('-b', '--bundle', type=str, required=False, default='sharadar', metavar='bundlename')
    trade.add_argument('-f', '--freq', type=str, required=False, default='minute', metavar='daily | minute')

    bundle.add_argument('-i', '--ingest', type=str, required=False, default='sharadar', metavar='bundlename')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    if args.command == 'backtest':
        run_backtest(args.strategyname, args.start, args.end, args.bundle, args.capital, args.freq)

    elif args.command == 'trade':
        run_trade(args.strategyname, args.twsuri, args.bundle, args.freq)

    elif args.command == 'bundle':

        load_extensions(
            default=True,
            extensions=[],
            strict=True,
            environ=os.environ,
        )
        
        ingest(args.ingest)
    
    elif args.command == 'calendar':

        if args.state == 'isopen':
            status = 0 if isopen_after_time(args.name, args.after) else 1
            sys.exit(status)

if __name__ == '__main__':
    main()