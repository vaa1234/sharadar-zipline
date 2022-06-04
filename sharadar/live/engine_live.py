import os
import pandas as pd
from zipline.data import bundles
import zipline.pipeline.domain as domain
from zipline.pipeline import Pipeline
from zipline.pipeline.loaders import EquityPricingLoader
from zipline.pipeline.data import EquityPricing
from zipline.pipeline.factors import Returns
from zipline.pipeline.engine import SimplePipelineEngine
from exchange_calendars import get_calendar
from sharadar.pipeline.engine import load_sharadar_bundle, CliProgressPublisher
from zipline.pipeline.loaders.equity_pricing_loader import USEquityPricingLoader
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.domain import US_EQUITIES
from pathlib import Path
from sharadar.util.logger import SharadarLogger
log = SharadarLogger(Path(__file__).stem)
from zipline.pipeline.loaders.equity_pricing_loader import shift_dates
from zipline.lib.adjusted_array import AdjustedArray
from zipline.utils.numpy_utils import repeat_first_axis
from zipline.pipeline.hooks.progress import ProgressHooks

def make_pipeline_live_engine(bundle=None, start=None, end=None, live=False):

    if bundle is None:
        bundle = load_sharadar_bundle()

    if start is None:
        start = bundle.equity_daily_bar_reader.first_trading_day

    if end is None:
        end = pd.to_datetime('today', utc=True)

    bundle.asset_finder.is_live_trading = live

    def load_adjusted_array_without_shift(self, domain, columns, dates, sids, mask):
            # in real live trading mode, we do not shift data on a day back
            sessions = domain.all_sessions()
            shifted_dates = shift_dates(sessions, dates[0], dates[-1], shift=0) # 0 - without shifting

            ohlcv_cols, currency_cols = self._split_column_types(columns)
            del columns  # From here on we should use ohlcv_cols or currency_cols.
            ohlcv_colnames = [c.name for c in ohlcv_cols]

            raw_ohlcv_arrays = self.raw_price_reader.load_raw_arrays(
                ohlcv_colnames,
                shifted_dates[0],
                shifted_dates[-1],
                sids,
            )

            # Currency convert raw_arrays in place if necessary. We use shifted
            # dates to load currency conversion rates to make them line up with
            # dates used to fetch prices.
            self._inplace_currency_convert(
                ohlcv_cols,
                raw_ohlcv_arrays,
                shifted_dates,
                sids,
            )

            adjustments = self.adjustments_reader.load_pricing_adjustments(
                ohlcv_colnames,
                dates,
                sids,
            )

            out = {}
            for c, c_raw, c_adjs in zip(ohlcv_cols, raw_ohlcv_arrays, adjustments):
                out[c] = AdjustedArray(
                    c_raw.astype(c.dtype),
                    c_adjs,
                    c.missing_value,
                )

            for c in currency_cols:
                codes_1d = self.raw_price_reader.currency_codes(sids)
                codes = repeat_first_axis(codes_1d, len(dates))
                out[c] = AdjustedArray(
                    codes,
                    adjustments={},
                    missing_value=None,
                )

            return out

    USEquityPricingLoader_live = USEquityPricingLoader
    USEquityPricingLoader_live.load_adjusted_array = load_adjusted_array_without_shift
    pipeline_loader = USEquityPricingLoader_live.without_fx(bundle.equity_daily_bar_reader, bundle.adjustment_reader)
    
    def choose_loader(column):
        if column in USEquityPricing.columns:
            return pipeline_loader
        raise ValueError("No PipelineLoader registered for column %s." % column)

    class LivePipelineEngine(SimplePipelineEngine):
        def __init__(self, get_loader, asset_finder, default_domain=US_EQUITIES, populate_initial_workspace=None,
                    default_hooks=None):
            super().__init__(get_loader, asset_finder, default_domain, populate_initial_workspace, default_hooks)

        def run_pipeline(self, pipeline, start_date, end_date=None, chunksize=1, hooks=None):
            if end_date is None:
                end_date = start_date

            if hooks is None:
                hooks = [ProgressHooks.with_static_publisher(CliProgressPublisher())]

            log.info("Compute pipeline values in live trading mode")
            return super().run_pipeline(pipeline, start_date, end_date, hooks)

    engine = LivePipelineEngine(get_loader=choose_loader, asset_finder=bundle.asset_finder)
    return engine