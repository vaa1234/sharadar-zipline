import alphalens as al
import pandas as pd
import numpy as np
import empyrical as em
from pandas.tseries.offsets import Day, BDay
import pyfolio as pf
from pyfolio.tears import utils
from pyfolio.utils import format_asset
from pyfolio import round_trips as rt
from pyfolio.plotting import STAT_FUNCS_PCT
import time
# backward compatible with quantrocket zipline
try:
    from sharadar.pipeline.engine import symbol, returns, prices as get_pricing
    from sharadar.util.run_algo import run_algorithm
    from sharadar.util.telegram import notify_telegram
except:
    pass
from IPython.display import display, HTML

import plotly
import plotly.graph_objects as go

from zipline.api import set_commission, order_target_percent
from zipline.finance.commission import PerShare

def analyze_zipline(strategy, benchmark=None):
    
    strategy_perf = pd.read_pickle(strategy)
    rets, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(strategy_perf)
    
    if benchmark is not None:
        benchmark_perf = pd.read_pickle(benchmark)
        benchmark_rets, benchmark_positions, benchmark_transactions = pf.utils.extract_rets_pos_txn_from_zipline(benchmark_perf)
    else:
        benchmark_rets = None
    
    display(HTML('Backtest name: {}'.format(strategy_perf.columns.name)))
    if benchmark is not None:
        display(HTML('Benchmark name: {}'.format(benchmark_perf.columns.name)))
    display(HTML('Start date: {}'.format(rets.index[0].strftime('%Y-%m-%d'))))
    display(HTML('End date: {}'.format(rets.index[-1].strftime('%Y-%m-%d'))))
    display(HTML('Total months: {}'.format(int(len(rets) / 21))))

    perf_stats_series = pf.timeseries.perf_stats(rets, positions=positions, transactions=transactions)

    perf_stats_df = pd.DataFrame(perf_stats_series, columns=['Backtest'])
    perf_stats_df = perf_stats_df.append(pd.DataFrame(index=['Transactions count'], data={'Backtest': len(transactions.index)}))
    perf_stats_df = perf_stats_df.append(pd.DataFrame(index=['Annual transaction costs'], data={'Backtest': annual_transaction_costs(strategy_perf)}))
    perf_stats_df = perf_stats_df.append(pd.DataFrame(index=['Total comissions amount'], data={'Backtest': total_comissions_amount(strategy_perf)}))

    if benchmark is not None:
        
        benckmarkperf_stats_series = pf.timeseries.perf_stats(benchmark_rets, positions=benchmark_positions, transactions=benchmark_transactions)
        perf_stats_df['Benchmark'] = benckmarkperf_stats_series

        perf_stats_df.at['Transactions count', 'Benchmark'] = len(benchmark_transactions.index)
        perf_stats_df.at['Annual transaction costs', 'Benchmark'] = annual_transaction_costs(benchmark_perf)
        perf_stats_df.at['Total comissions amount', 'Benchmark'] = total_comissions_amount(benchmark_perf)
        perf_stats_df.at['Correlation', 'Backtest'] = rets.corr(benchmark_rets)
        
        perf_stats_df['Spread'] = perf_stats_df['Backtest'] - perf_stats_df['Benchmark']

    for column in perf_stats_df.columns:
        for stat, value in perf_stats_df[column].iteritems():
            if stat in STAT_FUNCS_PCT + ['Annual transaction costs']:
                perf_stats_df.loc[stat, column] = str(np.round(value * 100, 2)) + '%'
                
    display(HTML(perf_stats_df.to_html()))
    
    display(HTML(holding_period_map(rets, benchmark_rets)))
    
    if benchmark is not None:
        corr_chart(rets, benchmark_rets)

    cumulative_return_chart(rets, benchmark_rets)
        
    pf.create_round_trip_tear_sheet(returns=rets, positions=positions, transactions=transactions, return_fig=True)
    
    pf.create_returns_tear_sheet(rets, positions, transactions, benchmark_rets=benchmark_rets, return_fig=True)
    
    pf.create_position_tear_sheet(rets, positions, return_fig=True)
    
    pf.create_txn_tear_sheet(rets, positions, transactions, return_fig=True)

    pf.create_interesting_times_tear_sheet(rets, return_fig=True)

def corr_chart(rets, benchmark_rets):

    corr = rets.rolling(30).corr(benchmark_rets)

    fig_corr = go.Figure(layout = go.Layout(yaxis = dict(tickformat=",.2f"),
                                                       xaxis = dict(tickformat= '%Y-%m-%d'),
                                                       hovermode="x unified",
                                                       legend_orientation="h",
                                                       yaxis_range=(-1, 1),
                                                       title_text="30 day correlation over time"
                                                      )
                                   )

    fig_corr.add_trace(
        go.Scatter(
            x=rets.index, 
            y=corr,
            hoverinfo='x+y',
            # name='Bonds',
            mode='lines',
            line=dict(width=0.5, color='rgb(131, 90, 241)'),
            # stackgroup='one' # define stack group
        )
    )

    return fig_corr.show()

def returns_analyze(strategy_returns, benchmark_returns=None):
    
    display(HTML('Backtest name: {}'.format(strategy_returns.name)))
    display(HTML('Start date: {}'.format(strategy_returns.index[0].strftime('%Y-%m-%d'))))
    display(HTML('End date: {}'.format(strategy_returns.index[-1].strftime('%Y-%m-%d'))))
    display(HTML('Total months: {}'.format(int(len(strategy_returns) / 21))))

    strategy_stats = pf.timeseries.perf_stats(strategy_returns)
    
    perf_stats_df = pd.DataFrame(strategy_stats, columns=['Backtest'])
    
    if benchmark_returns is not None:
        
        benchmark_stats = pf.timeseries.perf_stats(benchmark_returns)
        perf_stats_df['Benchmark'] = benchmark_stats
        perf_stats_df['Spread'] = perf_stats_df['Backtest'] - perf_stats_df['Benchmark']
        
    for column in perf_stats_df.columns:
        for stat, value in perf_stats_df[column].iteritems():
            if stat in STAT_FUNCS_PCT + ['Annual transaction costs']:
                perf_stats_df.loc[stat, column] = str(np.round(value * 100, 2)) + '%'
                
    display(HTML(perf_stats_df.to_html()))
    
    display(HTML(holding_period_map(strategy_returns, benchmark_returns)))
    
    if benchmark_returns is not None:
        cumulative_return_chart(strategy_returns, benchmark_returns)
        pf.create_returns_tear_sheet(strategy_returns, benchmark_rets=benchmark_returns, return_fig=True)
    else:
        cumulative_return_chart(strategy_returns)
        pf.create_returns_tear_sheet(strategy_returns, return_fig=True)

def cumulative_return_chart(rets, benchmark_rets=None):
    
    rets = (rets + 1).cumprod()
    # rets = em.cum_returns(rets)

    if benchmark_rets is not None:
        # benchmark_rets = em.cum_returns(benchmark_rets)
        benchmark_rets = (benchmark_rets + 1).cumprod()

    fig = go.Figure(layout = go.Layout(yaxis = dict(tickformat=",.2f"),
                                        xaxis = dict(tickformat= '%Y-%m-%d',
                                                    rangeslider_visible=True),
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                        ),
                                         hovermode="x unified",
                                         yaxis_type="log",
                                         title_text="Cumulative return (log scale)",
                                         height=800
                                        )
                     )
                         
    fig.add_trace(
        go.Scatter(
            x = rets.index,
            y = rets.values,
            mode="lines",
            name='Backtest'
        )
    )
    if benchmark_rets is not None:
        fig.add_trace(
            go.Scatter(
                x = benchmark_rets.index,
                y = benchmark_rets.values,
                mode="lines",
                name='Benchmark'
            )
        )

    return fig.show()

def assetallocation_chart(perf):

    fig_assetallocation = go.Figure(layout = go.Layout(yaxis = dict(tickformat=",.1%"),
                                                       xaxis = dict(tickformat= '%Y-%m-%d'),
                                                       hovermode="x unified",
                                                       legend_orientation="h",
                                                       yaxis_range=(0, 1),
                                                       title_text="Asset allocation over time"
                                                      )
                                   )

    fig_assetallocation.add_trace(
        go.Scatter(
            x=perf.index, 
            y=perf.bonds_weight,
            hoverinfo='x+y',
            name='Bonds',
            mode='lines',
            line=dict(width=0.5, color='rgb(131, 90, 241)'),
            stackgroup='one' # define stack group
        )
    )
    
    fig_assetallocation.add_trace(
        go.Scatter(
            x=perf.index, 
            y=perf.stocks_weight,
            hoverinfo='x+y',
            name='Stocks',
            mode='lines',
            line=dict(width=0.5, color='rgb(111, 231, 219)'),
            stackgroup='one'
        )
    )

    return fig_assetallocation.show()

def holding_period_map(strategy, benckmark=None):
    
    if benckmark is not None:

        benckmark_yr = em.aggregate_returns(benckmark, 'yearly')

    yr = em.aggregate_returns(strategy, 'yearly')
    df = pd.DataFrame(columns=range(1,len(yr)+1), index=yr.index)

    yr_start = 0
    
    if benckmark is not None:
        table = "<h3>Annual return holding period map and spread to benchmark</h3>"
    else:
        table = "<h3>Annual return holding period map</h3>"
    table += "<table class='table table-hover table-condensed table-striped'>"
    table += "<tr><th>Years</th>"

    for i in range(len(yr)):
        table += "<th>{}</th>".format(i+1)
    table += "</tr>"

    for the_year, value in yr.iteritems(): # Iterates years
        table += "<tr><th>{}</th>".format(the_year) # New table row

        for yrs_held in (range(1, len(yr)+1)): # Iterates yrs held 
            if yrs_held <= len(yr[yr_start:yr_start + yrs_held]):
                ret = em.annual_return(yr[yr_start:yr_start + yrs_held], 'yearly' )
                
                if benckmark is not None:
                    benckmark_ret = em.annual_return(benckmark_yr[yr_start:yr_start + yrs_held], 'yearly' )
                    diff_ret = ret - benckmark_ret
                    table += "<td>{:+.0f}({:+.0f})</td>".format(ret * 100, diff_ret * 100)
                else:
                    table += "<td>{:+.0f}</td>".format(ret * 100)
                    
        table += "</tr>"    
        yr_start+=1
    
    return table

def annual_transaction_costs(perf):
    
    perf['commission_amount'] = perf.loc[perf.orders.astype(str) != '[]']['orders'].apply(lambda x: sum([d['commission'] for d in x]))
    
    perf['commission_percent'] = perf['commission_amount'] / perf['portfolio_value']
    
    perf.commission_percent.fillna(0, inplace=True)
    
    annual_transaction_costs = em.annual_return(perf['commission_percent'])
    total_comissions_amount = perf.commission_amount.sum()
    
    return annual_transaction_costs

def total_comissions_amount(perf):
    
    perf['commission_amount'] = perf.loc[perf.orders.astype(str) != '[]']['orders'].apply(lambda x: sum([d['commission'] for d in x]))
    
    perf['commission_percent'] = perf['commission_amount'] / perf['portfolio_value']
    
    perf.commission_percent.fillna(0, inplace=True)
    
    annual_transaction_costs = em.annual_return(perf['commission_percent'])
    total_comissions_amount = int(perf.commission_amount.sum())
    
    return total_comissions_amount

def buyhold_return(sym, start_date, end_date):
        
    def initialize(context):
        
        set_commission(PerShare(cost=0.0035, min_trade_cost=0.35))
        
        context.has_ordered = False
        
    def handle_data(context, data):
        
        if not context.has_ordered:
            order_target_percent(symbol(sym), 1)
            context.has_ordered = True
        # order_target_percent(symbol(sym), 1)
    
    result = run_algorithm(
        handle_data=handle_data,
        start = start_date,
        end = end_date,
        initialize=initialize, # Define startup function
        capital_base=100000, # Set initial capital
        data_frequency = 'daily', # Set data frequency
        benchmark_symbol = None
    )
    
    returns, pos, trans = pf.utils.extract_rets_pos_txn_from_zipline(result)
    returns.name = sym

    return returns

def alphalens_quantile_plot(df, start_date, end_date, period):
    assets = df.index.levels[1].unique().tolist()
    df = df.dropna()
    start_price_date = start_date.strftime('%Y-%m-%d')
    end_price_date = (end_date + BDay(period)).strftime('%Y-%m-%d')
    pricing = get_pricing(
        assets,
        start_price_date,
        end_price_date,
        'close'
    )
    
    factor_names = df.columns
    factor_data = {}

    start_time = time.clock()
    for factor in factor_names:
        print("Formatting factor data for: " + factor)
        factor_data[factor] = al.utils.get_clean_factor_and_forward_returns(
            factor=df[factor],
            prices=pricing,
            periods=[period]
        )
    end_time = time.clock()
    print("Time to get arrange factor data: %.2f secs" % (end_time - start_time))
    
    qr_factor_returns = []

    for i, factor in enumerate(factor_names):
        mean_ret, _ = al.performance.mean_return_by_quantile(factor_data[factor])
        mean_ret.columns = [factor]
        qr_factor_returns.append(mean_ret)

    df_qr_factor_returns = pd.concat(qr_factor_returns, axis=1)

    return (10000*df_qr_factor_returns).plot.bar(
        subplots=True,
        sharey=True,
        layout=(4,2),
        figsize=(14, 14),
        legend=False,
        title='Alphas Comparison: Basis Points Per Day per Quantile'
    )
    
def alphalens_factor_plot(df, start_date, end_date, period):
    assets = df.index.levels[1].unique().tolist()
    df = df.dropna()
    start_price_date = start_date.strftime('%Y-%m-%d')
    end_price_date = (end_date + BDay(period)).strftime('%Y-%m-%d')
    pricing = get_pricing(
        assets,
        start_price_date,
        end_price_date,
        'close'
    )
    
    factor_names = df.columns
    factor_data = {}

    start_time = time.clock()
    for factor in factor_names:
        print("Formatting factor data for: " + factor)
        factor_data[factor] = al.utils.get_clean_factor_and_forward_returns(
            factor=df[factor],
            prices=pricing,
            periods=[period],
            quantiles=1
        )
    end_time = time.clock()
    print("Time to get arrange factor data: %.2f secs" % (end_time - start_time))
    
    ls_factor_returns = []

    start_time = time.clock()
    for i, factor in enumerate(factor_names):
        ls = al.performance.factor_returns(factor_data[factor])
        ls.columns = [factor]
        ls_factor_returns.append(ls)
    end_time = time.clock()
    print("Time to generate long/short returns: %.2f secs" % (end_time - start_time))

    df_ls_factor_returns = pd.concat(ls_factor_returns, axis=1)
    return (1+df_ls_factor_returns).cumprod().plot(title='Cumulative Factor Returns')

def strategy_daily_stats(perf, to_telegram=False):

    output = ("Daily stats of strategy name: {}".format(perf.columns.name))
    print(output)
    if to_telegram:
        notify_telegram(output)

    transactions = pd.DataFrame(data=perf.iloc[0]['transactions'])
    transactions.set_index('dt', inplace=True)
    transactions['sid'] = transactions['sid'].apply(lambda x: x.symbol)
    transactions = transactions[['amount', 'price', 'sid']]
    transactions = transactions.rename(columns={'sid': 'symbol'})
    trades = rt.extract_round_trips(transactions)

    output = ('Date: {}'.format(perf.index[0].strftime('%Y-%m-%d')))
    print(output)
    if to_telegram:
        notify_telegram(output)

    output = ('Algo return: {:.2%}'.format(perf.iloc[0]['algorithm_period_return']))
    print(output)
    if to_telegram:
        notify_telegram(output)
    
    output = ('Pnl, $: {:,.0f}'.format(perf.iloc[0]['pnl']))
    print(output)
    if to_telegram:
        notify_telegram(output)

    for index, trade in trades.iterrows():
        
        output = ('Symbol: {}, side: {}, return: {:.2%} ({:,.0f})'.format(trade.symbol, 'Short' if not trade.long else 'Long', trade.rt_returns, trade.pnl))
        print(output)
        if to_telegram:
            notify_telegram(output)