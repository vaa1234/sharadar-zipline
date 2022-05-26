#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This implementation uses IbPy, a third-party implementation of the API.
IbPy is not a product of Interactive Brokers, nor is this project affiliated with IB.
"""
import threading
import sys
from collections import namedtuple, defaultdict, OrderedDict
from time import sleep

from math import fabs

from sharadar.live.brokers.ib_execution import TWSOrder
from six import iteritems
import polling
import pandas as pd
import numpy as np

from sharadar.live.brokers.broker import Broker
from zipline.finance.order import Order as ZPOrder
from zipline.finance.order import ORDER_STATUS as ZP_ORDER_STATUS
from zipline.finance.execution import (MarketOrder,
                                       LimitOrder,
                                       StopOrder,
                                       StopLimitOrder)
from zipline.finance.transaction import Transaction
import zipline.protocol as zp
from zipline.protocol import MutableView
from zipline.api import symbol as symbol_lookup
from zipline.errors import SymbolNotFound
from ibapi.tag_value import TagValue
from ibapi.client import EClient
from ibapi.wrapper import EWrapper, TickTypeEnum
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.execution import ExecutionFilter

from sharadar.util.logger import log

if sys.version_info > (3,):
    long = int

IBPosition = namedtuple('IBPosition', ['contract', 'position', 'market_price',
                                       'market_value', 'average_cost',
                                       'unrealized_pnl', 'realized_pnl',
                                       'account_name'])

SHARADAR_TO_IB_EXCHANGES_MAP = {'BATS': 'BATS', 
                                'NASDAQ': 'NASDAQ', 
                                'NYSE': 'NYSE',
                                'NYSEARCA': 'ARCA',
                                'NYSEMKT': 'AMEX' 
                                }

_max_wait_subscribe = 10  # how many cycles to wait
_connection_timeout = 15  # Seconds
_poll_frequency = 0.1


def log_message(message, mapping):
    try:
        del (mapping['self'])
    except (KeyError,):
        pass
    items = list(mapping.items())
    items.sort()
    log.debug(('### %s' % (message,)))
    for k, v in items:
        log.debug(('    %s:%s' % (k, v)))


def _method_params_to_dict(args):
    return {k: v
            for k, v in iteritems(args)
            if k != 'self'}


class TWSConnection(EWrapper, EClient):
    def __init__(self, tws_uri):
        """
        :param tws_uri: host:listening_port:client_id
                        - host ip of running tws or ibgw
                        - port, default for tws 7496 and for ibgw 4002
                        - your client id, could be any number as long as it's not already used
        """
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.tws_uri = tws_uri
        host, port, client_id = self.tws_uri.split(':')
        self._host = host
        self._port = int(port)
        self.client_id = int(client_id)

        self._next_ticker_id = 0
        self._next_request_id = 0
        self._next_order_id = None
        self.managed_accounts = None
        self.symbol_to_ticker_id = {}
        self.ticker_id_to_symbol = {}
        self.last_tick = defaultdict(dict)
        self.bars = {}
        # accounts structure: accounts[account_id][currency][value]
        self.accounts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: np.NaN)))
        self.accounts_download_complete = False
        self.ib_positions = {}
        self.open_orders = {}
        self.order_statuses = {}
        self.executions = defaultdict(OrderedDict)
        self.commissions = defaultdict(OrderedDict)
        self._execution_to_order_id = {}
        self.time_skew = None
        self.unrecoverable_error = False
        self.tick_dict = {}

    def start(self):
        self.bind()

        # Initialise the threads for various components
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        setattr(self, "_thread", thread)

        timeout = _connection_timeout
        while timeout and not self.isConnected():
            log.info("Cannot connect to TWS. Retrying...")
            sleep(_poll_frequency)
            timeout -= _poll_frequency
        else:
            if not self.isConnected():
                raise SystemError("Connection timeout during TWS connection!")

        self._download_account_details()
        log.info("Managed accounts: {}".format(self.managed_accounts))

        self.reqCurrentTime()
        self.reqIds(1)

        while self.time_skew is None or self._next_order_id is None:
            sleep(_poll_frequency)

        log.info("Local-Broker Time Skew: {}".format(self.time_skew))

    def bind(self):
        log.info("Connecting: {}:{}:{}".format(self._host, self._port, self.client_id))
        self.connect(self._host, self._port, self.client_id)

    def _download_account_details(self):
        exec_filter = ExecutionFilter()
        exec_filter.clientId = self.client_id
        self.reqExecutions(self.next_request_id, exec_filter)

        self.reqManagedAccts()
        while self.managed_accounts is None:
            sleep(_poll_frequency)

        for account in self.managed_accounts:
            self.reqAccountUpdates(subscribe=True, acctCode=account)
        while self.accounts_download_complete is False:
            sleep(_poll_frequency)

    @property
    def next_ticker_id(self):
        ticker_id = self._next_ticker_id
        self._next_ticker_id += 1
        return ticker_id

    @property
    def next_request_id(self):
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    @property
    def next_order_id(self):
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    def subscribe_to_market_data(self, ib_symbol, exchange='SMART', primaryExchange='ISLAND', secType='STK', currency='USD'):
        if ib_symbol in self.symbol_to_ticker_id:
            # Already subscribed to market data
            return

        contract = Contract()
        contract.symbol = ib_symbol
        contract.exchange = exchange
        contract.primaryExchange = primaryExchange
        contract.secType = secType
        contract.currency = currency
        ticker_id = self.next_ticker_id

        self.symbol_to_ticker_id[ib_symbol] = ticker_id
        self.ticker_id_to_symbol[ticker_id] = ib_symbol

        # INDEX tickers cannot be requested with market data. The data can,
        # however, be requested with realtimeBars. This change will make
        # sure we can request data from INDEX tickers like SPX, VIX, etc.
        if contract.secType == 'IND':
            self.reqRealTimeBars(ticker_id, contract, 60, 'TRADES', True, [])
        else:
            self.reqMktData(ticker_id, contract, "", True, False, [])
            sleep(0.9) # default 11

    def _process_tick(self, ticker_id, tick_type, value):
        try:
            symbol = self.ticker_id_to_symbol[ticker_id]
        except KeyError:
            log.error("Tick {} for id={} is not registered".format(tick_type, ticker_id))
            return

        self.tick_dict["SYMBOL_" + str(ticker_id)] = symbol
        self.tick_dict[TickTypeEnum.to_str(tick_type) + "_" + str(ticker_id)] = value

    def tickSnapshotEnd(self, ticker_id):
        key = "SYMBOL_" + str(ticker_id)
        if key not in self.tick_dict:
            return

        ib_symbol = self.tick_dict[key]
        try:
            last_trade_price = float(self.tick_dict["LAST_" + str(ticker_id)])
            last_trade_size = int(self.tick_dict["LAST_SIZE_" + str(ticker_id)])
            last_trade_time = float(self.tick_dict["LAST_TIMESTAMP_" + str(ticker_id)])
            last_trade_dt = pd.to_datetime(last_trade_time, unit='s', utc=True)
        except KeyError:
            log.warning('Cannot subscribe market data for %s.' % ib_symbol)
            return

        self._add_bar(ib_symbol, last_trade_price, last_trade_size, last_trade_dt)

    def _add_bar(self, symbol, last_trade_price, last_trade_size, last_trade_time):
        bar = pd.DataFrame(index=pd.DatetimeIndex([last_trade_time]),
                           data={'last_trade_price': last_trade_price,
                                 'last_trade_size': last_trade_size})

        if symbol not in self.bars:
            self.bars[symbol] = bar
        else:
            self.bars[symbol] = self.bars[symbol].append(bar)

    def tickPrice(self, ticker_id, field, price, can_auto_execute):
        self._process_tick(ticker_id, tick_type=field, value=price)

    def tickSize(self, ticker_id, field, size):
        self._process_tick(ticker_id, tick_type=field, value=size)

    def tickOptionComputation(self, ticker_id, field, tick_attrib, implied_vol, delta, opt_price, pv_dividend, gamma,
                              vega,
                              theta, und_price):
        log_message('tickOptionComputation', vars())

    def tickGeneric(self, ticker_id, tick_type, value):
        self._process_tick(ticker_id, tick_type=tick_type, value=value)

    def tickString(self, ticker_id, tick_type, value):
        self._process_tick(ticker_id, tick_type=tick_type, value=value)

    def tickEFP(self, ticker_id, tick_type, basis_points,
                formatted_basis_points, implied_future, hold_days,
                future_expiry, dividend_impact, dividends_to_expiry):
        log_message('tickEFP', vars())

    def updateAccountValue(self, key, value, currency, account_name):
        self.accounts[account_name][currency][key] = value

    def updatePortfolio(self,
                        contract,
                        position,
                        market_price,
                        market_value,
                        average_cost,
                        unrealized_pnl,
                        realized_pnl,
                        account_name):
        symbol = contract.symbol

        ib_position = IBPosition(contract=contract,
                                 position=position,
                                 market_price=market_price,
                                 market_value=market_value,
                                 average_cost=average_cost,
                                 unrealized_pnl=unrealized_pnl,
                                 realized_pnl=realized_pnl,
                                 account_name=account_name)

        self.ib_positions[symbol] = ib_position

    def updateAccountTime(self, time_stamp):
        pass

    def accountDownloadEnd(self, account_name):
        self.accounts_download_complete = True

    def nextValidId(self, order_id):
        self._next_order_id = order_id

    def contractDetails(self, req_id, contract_details):
        log_message('contractDetails', vars())

    def contractDetailsEnd(self, req_id):
        log_message('contractDetailsEnd', vars())

    def bondContractDetails(self, req_id, contract_details):
        log_message('bondContractDetails', vars())

    def orderStatus(self, order_id, status, filled, remaining, avg_fill_price, perm_id, parent_id, last_fill_price,
                    client_id, why_held, mkt_cap):
        self.order_statuses[order_id] = _method_params_to_dict(vars())

        log.debug(
            "Order-{order_id} {status}: "
            "filled={filled} remaining={remaining} "
            "avg_fill_price={avg_fill_price} "
            "last_fill_price={last_fill_price} ".format(
                order_id=order_id,
                status=self.order_statuses[order_id]['status'],
                filled=self.order_statuses[order_id]['filled'],
                remaining=self.order_statuses[order_id]['remaining'],
                avg_fill_price=self.order_statuses[order_id]['avg_fill_price'],
                last_fill_price=self.order_statuses[order_id]['last_fill_price']))

    def openOrder(self, order_id, contract, order, state):
        self.open_orders[order_id] = _method_params_to_dict(vars())

        log.debug(
            "Order-{order_id} {status}: "
            "{order_action} {order_count} {symbol} with {order_type} order. "
            "limit_price={limit_price} stop_price={stop_price}".format(
                order_id=order_id,
                status=state.status,
                order_action=order.action,
                order_count=order.totalQuantity,
                symbol=contract.symbol,
                order_type=order.orderType,
                limit_price=order.lmtPrice,
                stop_price=order.auxPrice))

    def openOrderEnd(self):
        pass

    def execDetails(self, req_id, contract, exec_detail):
        order_id, exec_id = exec_detail.orderId, exec_detail.execId
        self.executions[order_id][exec_id] = _method_params_to_dict(vars())
        self._execution_to_order_id[exec_id] = order_id

        log.info(
            "Order-{order_id} executed @ {exec_time}: "
            "{symbol} current: {shares} @ ${price} "
            "total: {cum_qty} @ ${avg_price} "
            "exec_id: {exec_id} by client-{client_id}".format(
                order_id=order_id, exec_id=exec_id,
                exec_time=pd.to_datetime(exec_detail.time),
                symbol=contract.symbol,
                shares=exec_detail.shares,
                price=exec_detail.price,
                cum_qty=exec_detail.cumQty,
                avg_price=exec_detail.avgPrice,
                client_id=exec_detail.clientId))

    def execDetailsEnd(self, req_id):
        log.debug(
            "Execution details completed for request {req_id}".format(
                req_id=req_id))

    def commissionReport(self, commission_report):
        exec_id = commission_report.execId
        order_id = self._execution_to_order_id[commission_report.execId]
        self.commissions[order_id][exec_id] = commission_report

        log.debug(
            "Order-{order_id} report: "
            "realized_pnl: ${realized_pnl} "
            "commission: ${commission} yield: {yield_} "
            "exec_id: {exec_id}".format(
                order_id=order_id,
                exec_id=commission_report.execId,
                realized_pnl=commission_report.realizedPNL
                if commission_report.realizedPNL != sys.float_info.max
                else 0,
                commission=commission_report.commission,
                yield_=commission_report.yield_
                if commission_report.yield_ != sys.float_info.max
                else 0)
        )

    def connectionClosed(self):
        self.unrecoverable_error = True
        log.info("IB Connection closed")

    def error(self, id_=None, error_code=None, error_msg=None):
        if isinstance(id_, Exception):
            log.exception(id_)

        if isinstance(error_code, int):
            if error_code in (502, 503, 326):
                # 502: Couldn't connect to TWS.
                # 503: The TWS is out of date and must be upgraded.
                # 326: Unable connect as the client id is already in use.
                self.unrecoverable_error = True

            if error_code < 1000:
                log.error("[{}] {} ({})".format(error_code, error_msg, id_))
            else:
                log.info("[{}] {} ({})".format(error_code, error_msg, id_))
        else:
            log.error("[{}] {} ({})".format(error_code, error_msg, id_))

    def updateMktDepth(self, ticker_id, position, operation, side, price,
                       size):
        log_message('updateMktDepth', vars())

    def updateMktDepthL2(self, ticker_id, position, market_maker, operation, side, price, size, is_smart_depth):
        log_message('updateMktDepthL2', vars())

    def updateNewsBulletin(self, msg_id, msg_type, message, orig_exchange):
        log_message('updateNewsBulletin', vars())

    def managedAccounts(self, accounts_list):
        self.managed_accounts = accounts_list.split(',')

    def receiveFA(self, fa_data_type, xml):
        log_message('receiveFA', vars())

    def historicalData(self, req_id, bar):
        log_message('historicalData', vars())

    def scannerParameters(self, xml):
        log_message('scannerParameters', vars())

    def scannerData(self, req_id, rank, contract_details, distance, benchmark,
                    projection, legs_str):
        log_message('scannerData', vars())

    def currentTime(self, time):
        self.time_skew = (pd.to_datetime('now', utc=True) -
                          pd.to_datetime(long(time), unit='s', utc=True))

    def deltaNeutralValidation(self, req_id, under_comp):
        log_message('deltaNeutralValidation', vars())

    def fundamentalData(self, req_id, data):
        log_message('fundamentalData', vars())

    def marketDataType(self, req_id, market_data_type):
        pass

    def realtimeBar(self, req_id, time, open_, high, low, close, volume, wap,
                    count):
        value = (";".join([str(close), str(count), str(time), str(volume),
                           str(wap), "true"]))
        self._process_tick(req_id, tick_type=48, value=value)

    def scannerDataEnd(self, req_id):
        log_message('scannerDataEnd', vars())

    def position(self, account, contract, pos, avg_cost):
        log_message('position', vars())

    def positionEnd(self):
        log_message('positionEnd', vars())

    def accountSummary(self, req_id, account, tag, value, currency):
        log_message('accountSummary', vars())

    def accountSummaryEnd(self, req_id):
        log_message('accountSummaryEnd', vars())


class IBBroker(Broker):
    def __init__(self, tws_uri, account_id=None):
        """
        :param tws_uri: host:listening_port:client_id
                        - host ip of running tws or ibgw
                        - port, default for tws 7496 and for ibgw 4002
                        - your client id, could be any number as long as it's not already used
        """
        self._tws_uri = tws_uri
        self._orders = {}
        self._transactions = {}

        self._tws = TWSConnection(tws_uri)
        self._tws.start()

        self.account_id = (self._tws.managed_accounts[0] if account_id is None
                           else account_id)
        self.currency = 'USD'

        self._subscribed_assets = []
        self.metrics_tracker = None

        super(self.__class__, self).__init__()

    @property
    def subscribed_assets(self):
        return self._subscribed_assets

    def disconnect(self):
        self._tws.disconnect()

    def subscribe_to_market_data(self, asset):
        if asset not in self.subscribed_assets:
            ib_symbol = self._asset_symbol(asset)
            exchange = 'SMART'
            primaryExchange = SHARADAR_TO_IB_EXCHANGES_MAP[asset.exchange]
            secType = 'STK'
            currency = 'USD'

            self._tws.subscribe_to_market_data(ib_symbol, exchange, primaryExchange, secType, currency)
            self._subscribed_assets.append(asset)
            try:
                polling.poll(
                    lambda: ib_symbol in self._tws.bars,
                    timeout=_max_wait_subscribe,
                    step=_poll_frequency)
            except polling.TimeoutException as te:
                log.warning('Cannot subscribe market data for %s.' % ib_symbol)
            else:
                log.debug("Subscription completed")

    @property
    def positions(self):
        self._get_positions_from_broker()
        # Filter out positions with amount == 0
        return {v : k for k, v in self.metrics_tracker.positions.items() if v.amount != 0}

    def _get_positions_from_broker(self):
        """
        get the positions from the broker and update zipline objects ( the ledger )
        should be used once at startup and once every time we want to refresh the positions array
        """
        cur_pos_in_tracker = self.metrics_tracker.positions
        for ib_symbol in self._tws.ib_positions:
            ib_position = self._tws.ib_positions[ib_symbol]
 
            equity = self._safe_symbol_lookup(ib_symbol)
            if not equity:
                log.warning('Wanted to subscribe to %s, but this asset is probably not ingested' % ib_symbol)
                continue

            zp_position = zp.Position(zp.InnerPosition(equity))
            editable_position = MutableView(zp_position)

            editable_position._underlying_position.amount = int(ib_position.position)
            editable_position._underlying_position.cost_basis = float(ib_position.average_cost)
            editable_position._underlying_position.last_sale_price = ib_position.market_price
            last_close = self.metrics_tracker._trading_calendar.session_close(self.metrics_tracker._last_session)
            editable_position._underlying_position.last_sale_date = last_close

            self.metrics_tracker.update_position(zp_position.asset,
                                                 amount=zp_position.amount,
                                                 last_sale_price=zp_position.last_sale_price,
                                                 last_sale_date=zp_position.last_sale_date,
                                                 cost_basis=zp_position.cost_basis)
        for asset in cur_pos_in_tracker:
            ib_symbol = self._asset_symbol(asset)
            if ib_symbol not in self._tws.ib_positions:
                # deleting object from the metrcs_tracker as its not in the portfolio
                self.metrics_tracker.update_position(asset, amount=0)

        self.metrics_tracker._ledger._portfolio.positions = zp.Positions(self.metrics_tracker.positions)

    @property
    def portfolio(self):
        self.positions  # update positions
        return self.metrics_tracker.portfolio

    def get_account_from_broker(self):
        ib_account = self._tws.accounts[self.account_id][self.currency]
        return ib_account

    def set_metrics_tracker(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker

    @property
    def account(self):
        ib_account = self._tws.accounts[self.account_id][self.currency]

        self.metrics_tracker.override_account_fields(
            settled_cash=float(ib_account['CashBalance']),
            accrued_interest=float(ib_account['AccruedCash']),
            buying_power=float(ib_account['BuyingPower']),
            equity_with_loan=float(ib_account['EquityWithLoanValue']),
            total_positions_value=float(ib_account['StockMarketValue']),
            total_positions_exposure=float(
                (float(ib_account['StockMarketValue']) /
                 (float(ib_account['StockMarketValue']) +
                  float(ib_account['TotalCashValue'])))),
            regt_equity=float(ib_account['RegTEquity']),
            regt_margin=float(ib_account['RegTMargin']),
            initial_margin_requirement=float(
                ib_account['FullInitMarginReq']),
            maintenance_margin_requirement=float(
                ib_account['FullMaintMarginReq']),
            available_funds=float(ib_account['AvailableFunds']),
            excess_liquidity=float(ib_account['ExcessLiquidity']),
            cushion=float(
                self._tws.accounts[self.account_id]['']['Cushion']),
            day_trades_remaining=float(
                self._tws.accounts[self.account_id]['']['DayTradesRemaining']),
            leverage=float(
                self._tws.accounts[self.account_id]['']['Leverage-S']),
            net_leverage=(
                    float(ib_account['StockMarketValue']) /
                    (float(ib_account['TotalCashValue']) +
                     float(ib_account['StockMarketValue']))),
            net_liquidation=float(ib_account['NetLiquidation'])
        )

        return self.metrics_tracker.account

    @property
    def time_skew(self):
        return self._tws.time_skew

    def is_alive(self):
        return not self._tws.unrecoverable_error

    def _to_ib_symbol(self, symbol):
        # some example: NAV-PD -> NAV, LGF.B -> LGF B
        return symbol.replace('.', ' ').partition('-')[0]

    def _asset_symbol(self, asset):
        return self._to_ib_symbol(str(asset.symbol))

    @staticmethod
    def _safe_symbol_lookup(ib_symbol):
        try:
            symbol = ib_symbol.replace(' ', '.')
            return symbol_lookup(symbol)
        except SymbolNotFound:
            return None

    _zl_order_ref_magic = '!ZL'

    @classmethod
    def _create_order_ref(cls, ib_order, dt=pd.to_datetime('now', utc=True)):
        order_type = ib_order.orderType.replace(' ', '_')
        return \
            "A:{action} Q:{qty} T:{order_type} " \
            "L:{limit_price} S:{stop_price} D:{date} {magic}".format(
                action=ib_order.action,
                qty=ib_order.totalQuantity,
                order_type=order_type,
                limit_price=ib_order.lmtPrice,
                stop_price=ib_order.auxPrice,
                date=int(dt.value / 1e9),
                magic=cls._zl_order_ref_magic)

    @classmethod
    def _parse_order_ref(cls, ib_order_ref):
        if not ib_order_ref or \
                not ib_order_ref.endswith(cls._zl_order_ref_magic):
            return None

        try:
            action, qty, order_type, limit_price, stop_price, dt, _ = \
                ib_order_ref.split(' ')

            if not all(
                    [action.startswith('A:'),
                     qty.startswith('Q:'),
                     order_type.startswith('T:'),
                     limit_price.startswith('L:'),
                     stop_price.startswith('S:'),
                     dt.startswith('D:')]):
                return None

            return {
                'action': action[2:],
                'qty': int(qty[2:]),
                'order_type': order_type[2:].replace('_', ' '),
                'limit_price': float(limit_price[2:]),
                'stop_price': float(stop_price[2:]),
                'dt': pd.to_datetime(dt[2:], unit='s', utc=True)}

        except ValueError:
            log.warning("Error parsing order metadata: {}".format(
                ib_order_ref))
            return None

    def order(self, asset, amount, style):
        ib_symbol = self._asset_symbol(asset)

        contract = Contract()
        contract.symbol = ib_symbol
        contract.exchange = 'SMART' # style.exchange if style.exchange is not None else 'SMART'
        contract.primaryExchange = SHARADAR_TO_IB_EXCHANGES_MAP[asset.exchange] # map sharadar asset primary exchange name to IB exchnages name
        contract.secType = 'STK'
        contract.currency = 'USD'

        order = Order()
        order.totalQuantity = int(fabs(amount))
        order.action = "BUY" if amount > 0 else "SELL"

        is_buy = (amount > 0)
        order.lmtPrice = style.get_limit_price(is_buy) or 0
        order.auxPrice = style.get_stop_price(is_buy) or 0

        if isinstance(style, MarketOrder):
            order.orderType = "MKT"
            # order.tif = "GTC"
            order.tif = "DAY"
            order.algoStrategy = "Adaptive"
            order.algoParams = []
            order.algoParams.append(TagValue("adaptivePriority", "Normal"))
        elif isinstance(style, LimitOrder):
            order.orderType = "LMT"
            order.tif = "GTC"
        elif isinstance(style, StopOrder):
            order.orderType = "STP"
            order.tif = "GTC"
        elif isinstance(style, StopLimitOrder):
            order.orderType = "STP LMT"
            order.tif = "GTC"
        elif isinstance(style, TWSOrder):
            order.orderType = style.get_order_type()
            order.tif = style.get_time_in_force()

        order.orderRef = self._create_order_ref(order)

        ib_order_id = self._tws.next_order_id
        zp_order = self._get_or_create_zp_order(ib_order_id, order, contract)

        self.log_order(contract, ib_order_id, order)

        self._tws.placeOrder(ib_order_id, contract, order)

        return zp_order

    def log_order(self, contract, ib_order_id, order):
        if order.orderType == "MKT":
            log.info(
                "Placing order-{order_id}: "
                "{action} {qty} {symbol} with MKT order {tif}.".format(
                    order_id=ib_order_id,
                    action=order.action,
                    qty=order.totalQuantity,
                    symbol=contract.symbol,
                    tif=order.tif
                ))
        else:
            log.info(
                "Placing order-{order_id}: "
                "{action} {qty} {symbol} with {order_type} order. "
                "limit_price={limit_price} stop_price={stop_price} {tif}".format(
                    order_id=ib_order_id,
                    action=order.action,
                    qty=order.totalQuantity,
                    symbol=contract.symbol,
                    order_type=order.orderType,
                    limit_price=order.lmtPrice,
                    stop_price=order.auxPrice,
                    tif=order.tif
                ))

    @property
    def orders(self):
        self._update_orders()
        return self._orders

    def _ib_to_zp_order_id(self, ib_order_id):
        return "IB-{date}-{account_id}-{client_id}-{order_id}".format(
            date=str(pd.to_datetime('today').date()),
            account_id=self.account_id,
            client_id=self._tws.client_id,
            order_id=ib_order_id)

    @staticmethod
    def _action_qty_to_amount(action, qty):
        return qty if action == 'BUY' else -1 * qty

    def _get_or_create_zp_order(self, ib_order_id,
                                ib_order=None, ib_contract=None):
        zp_order_id = self._ib_to_zp_order_id(ib_order_id)
        if zp_order_id in self._orders:
            return self._orders[zp_order_id]

        # Try to reconstruct the order from the given information:
        # open order state and execution state
        ib_symbol, order_details = None, None

        if ib_order and ib_contract:
            ib_symbol = ib_contract.symbol
            order_details = self._parse_order_ref(ib_order.orderRef)

        if not order_details and ib_order_id in self._tws.open_orders:
            open_order = self._tws.open_orders[ib_order_id]
            ib_symbol = open_order['contract'].symbol
            order_details = self._parse_order_ref(
                open_order['order'].orderRef)

        if not order_details and ib_order_id in self._tws.executions:
            executions = self._tws.executions[ib_order_id]
            last_exec_detail = list(executions.values())[-1]['exec_detail']
            last_exec_contract = list(executions.values())[-1]['contract']
            ib_symbol = last_exec_contract.symbol
            order_details = self._parse_order_ref(last_exec_detail.orderRef)

        asset = self._safe_symbol_lookup(ib_symbol)
        if not asset:
            log.warning(
                "Ignoring symbol {symbol} which has associated "
                "order but it is not registered in bundle".format(
                    symbol=ib_symbol))
            return None

        if order_details:
            amount = self._action_qty_to_amount(order_details['action'],
                                                order_details['qty'])
            stop_price = order_details['stop_price']
            limit_price = order_details['limit_price']
            dt = order_details['dt']
        else:
            dt = pd.to_datetime('now', utc=True)
            amount, stop_price, limit_price = 0, None, None
            if ib_order_id in self._tws.open_orders:
                open_order = self._tws.open_orders[ib_order_id]['order']
                amount = self._action_qty_to_amount(
                    open_order.action, open_order.totalQuantity)
                stop_price = open_order.auxPrice
                limit_price = open_order.lmtPrice

        stop_price = None if stop_price == 0 else stop_price
        limit_price = None if limit_price == 0 else limit_price

        self._orders[zp_order_id] = ZPOrder(
            dt=dt,
            asset=asset,
            amount=amount,
            stop=stop_price,
            limit=limit_price,
            id=zp_order_id)
        self._orders[zp_order_id].broker_order_id = ib_order_id

        return self._orders[zp_order_id]

    @staticmethod
    def _ib_to_zp_status(ib_status):
        ib_status = ib_status.lower()
        if ib_status == 'submitted':
            return ZP_ORDER_STATUS.OPEN
        elif ib_status in ('pendingsubmit',
                           'pendingcancel',
                           'presubmitted'):
            return ZP_ORDER_STATUS.HELD
        elif ib_status == 'cancelled':
            return ZP_ORDER_STATUS.CANCELLED
        elif ib_status == 'filled':
            return ZP_ORDER_STATUS.FILLED
        elif ib_status == 'inactive':
            return ZP_ORDER_STATUS.REJECTED
        else:
            return None

    def _update_orders(self):
        def _update_from_order_status(zp_order, ib_order_id):
            if ib_order_id in self._tws.open_orders:
                open_order_state = self._tws.open_orders[ib_order_id]['state']

                zp_status = self._ib_to_zp_status(open_order_state.status)
                if zp_status is None:
                    log.warning(
                        "Order-{order_id}: "
                        "unknown order status: {order_status}.".format(
                            order_id=ib_order_id,
                            order_status=open_order_state.status))
                else:
                    zp_order.status = zp_status

            if ib_order_id in self._tws.order_statuses:
                order_status = self._tws.order_statuses[ib_order_id]

                zp_order.filled = order_status['filled']

                zp_status = self._ib_to_zp_status(order_status['status'])
                if zp_status is None:
                    log.warning("Order-{order_id}: "
                                "unknown order status: {order_status}.".format(
                        order_id=ib_order_id,
                        order_status=order_status['status']))
                else:
                    zp_order.status = zp_status

        def _update_from_execution(zp_order, ib_order_id):
            if ib_order_id in self._tws.executions and \
                    ib_order_id not in self._tws.open_orders:
                zp_order.status = ZP_ORDER_STATUS.FILLED
                executions = self._tws.executions[ib_order_id]
                last_exec_detail = \
                    list(executions.values())[-1]['exec_detail']
                zp_order.filled = last_exec_detail.cumQty

        all_ib_order_ids = (set([e.broker_order_id
                                 for e in self._orders.values()]) |
                            set(self._tws.open_orders.keys()) |
                            set(self._tws.order_statuses.keys()) |
                            set(self._tws.executions.keys()) |
                            set(self._tws.commissions.keys()))
        for ib_order_id in all_ib_order_ids:
            zp_order = self._get_or_create_zp_order(ib_order_id)
            if zp_order:
                _update_from_execution(zp_order, ib_order_id)
                _update_from_order_status(zp_order, ib_order_id)

    @property
    def transactions(self):
        self._update_transactions()
        return self._transactions

    def _update_transactions(self):
        all_orders = list(self.orders.values())

        for ib_order_id, executions in iteritems(self._tws.executions):
            orders = [order
                      for order in all_orders
                      if order.broker_order_id == ib_order_id]

            if not orders:
                log.warning("No order found for executions: {}".format(executions))
                continue

            assert len(orders) == 1
            order = orders[0]

            for exec_id, execution in iteritems(executions):
                if exec_id in self._transactions:
                    continue

                try:
                    commission = self._tws.commissions[ib_order_id][exec_id].commission
                except KeyError:
                    log.warning(
                        "Commission not found for execution: {}".format(exec_id))
                    commission = 0

                exec_detail = execution['exec_detail']
                is_buy = order.amount > 0
                amount = (exec_detail.shares if is_buy
                          else -1 * exec_detail.shares)
                tx = Transaction(
                    asset=order.asset,
                    amount=amount,
                    dt=pd.to_datetime(exec_detail.time, utc=True),
                    price=exec_detail.price,
                    order_id=order.id
                )
                self._transactions[exec_id] = tx

    def cancel_order(self, zp_order_id):
        ib_order_id = self.orders[zp_order_id].broker_order_id
        self._tws.cancelOrder(ib_order_id)

    def get_spot_value(self, asset, field, dt, data_frequency):
        self.subscribe_to_market_data(asset)

        ib_symbol = self._asset_symbol(asset)
        bars = self._tws.bars[ib_symbol]

        last_event_time = bars.index[-1]

        minute_start = (last_event_time - pd.Timedelta('1 min')).time()
        minute_end = last_event_time.time()

        if bars.empty:
            return pd.NaT if field == 'last_traded' else np.NaN
        else:
            if field == 'price':
                return bars.last_trade_price.iloc[-1]
            elif field == 'last_traded':
                return last_event_time or pd.NaT

            minute_df = bars.between_time(minute_start, minute_end,
                                          include_start=True, include_end=True)
            if minute_df.empty:
                return np.NaN
            else:
                if field == 'open':
                    return minute_df.last_trade_price.iloc[0]
                elif field == 'close':
                    return minute_df.last_trade_price.iloc[-1]
                elif field == 'high':
                    return minute_df.last_trade_price.max()
                elif field == 'low':
                    return minute_df.last_trade_price.min()
                elif field == 'volume':
                    return minute_df.last_trade_size.sum()

    def get_last_traded_dt(self, asset):
        self.subscribe_to_market_data(asset)
        ib_symbol = self._asset_symbol(asset)
        return self._tws.bars[ib_symbol].index[-1]

    def get_realtime_bars(self, assets, frequency):
        if frequency == '1m':
            resample_freq = '1 Min'
        elif frequency == '1d':
            resample_freq = '24 H'
        else:
            raise ValueError("Invalid frequency specified: %s" % frequency)

        df = pd.DataFrame()
        for asset in assets:
            self.subscribe_to_market_data(asset)

            ib_symbol = self._asset_symbol(asset)
            if ib_symbol in self._tws.bars:
                trade_prices = self._tws.bars[ib_symbol]['last_trade_price']
                trade_sizes = self._tws.bars[ib_symbol]['last_trade_size']
            else:
                log.warning("No tws bar for '%s'." % ib_symbol)
                trade_prices = pd.Series([1.0], index=[pd.to_datetime('now')])
                trade_sizes = pd.Series([1], index=[pd.to_datetime('now')])
            ohlcv = trade_prices.resample(resample_freq).ohlc()
            ohlcv['volume'] = trade_sizes.resample(resample_freq).sum()

            # Add asset as level 0 column; ohlcv will be used as level 1 cols
            ohlcv.columns = pd.MultiIndex.from_product([[asset, ], ohlcv.columns])

            df = pd.concat([df, ohlcv], axis=1)

        return df





