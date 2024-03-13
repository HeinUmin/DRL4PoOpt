from datetime import datetime, date
import pandas as pd
import requests
from akshare.utils import demjson
import empyrical as ep
import scipy.stats as stats
import seaborn
from functools import wraps
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict
import numpy as np
from IPython.display import display, HTML
from matplotlib.ticker import FuncFormatter
from functools import partial
from config import config


def backtest_strat(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    ts = pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)
    return ts


# Modded from akshare
def fund_open_fund_rank_em(
    symbol: str = "全部",
    start_date: str = "",
    end_date: str = datetime.now().date().isoformat(),
) -> pd.DataFrame:
    """
    东方财富网-数据中心-开放基金排行
    https://fund.eastmoney.com/data/fundranking.html
    :param symbol: choice of {"全部", "股票型", "混合型", "债券型", "指数型", "QDII", "LOF", "FOF"}
    :type symbol: str
    :param start_date: 开始日期, 默认为空字符串
    :type start_date: str
    :param end_date: 结束日期, 默认为当天日期
    :type end_date: str
    :return: 开放基金排行
    :rtype: pandas.DataFrame
    """
    if start_date == "":
        date_str = end_date.replace("-", "")
        # 将字符串格式的日期转换为date对象
        given_date = date(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]))
        try:
            # 尝试直接设置为前一年，保持相同的月和日
            one_year_before = given_date.replace(year=given_date.year - 1)
        except ValueError:
            # 如果前一年没有相同的月和日（比如2月29日），则设置为2月28日
            one_year_before = given_date.replace(year=given_date.year - 1, day=28)
        start_date = one_year_before.isoformat()

    url = "https://fund.eastmoney.com/data/rankhandler.aspx"
    type_map = {
        "全部": ["all", "1nzf"],
        "股票型": ["gp", "1nzf"],
        "混合型": ["hh", "1nzf"],
        "债券型": ["zq", "1nzf"],
        "指数型": ["zs", "1nzf"],
        "QDII": ["qdii", "1nzf"],
        "LOF": ["lof", "1nzf"],
        "FOF": ["fof", "1nzf"],
    }
    params = {
        "op": "ph",
        "dt": "kf",
        "ft": type_map[symbol][0],
        "rs": "",
        "gs": "0",
        "sc": type_map[symbol][1],
        "st": "desc",
        "sd": start_date,
        "ed": end_date,
        "qdii": "",
        "tabSubtype": ",,,,,",
        "pi": "1",
        "pn": "20000",
        "dx": "1",
        "v": "0.1591891419018292",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
        "Referer": "https://fund.eastmoney.com/fundguzhi.html",
    }
    r = requests.get(url, params=params, headers=headers)
    data_text = r.text
    data_json = demjson.decode(data_text[data_text.find("{") : -1])
    temp_df = pd.DataFrame(data_json["datas"])
    temp_df = temp_df.iloc[:, 0].str.split(",", expand=True)
    temp_df.reset_index(inplace=True)
    temp_df["index"] = list(range(1, len(temp_df) + 1))
    temp_df.columns = [
        "序号",
        "基金代码",
        "基金简称",
        "拼音缩写",
        "日期",
        "单位净值",
        "累计净值",
        "日增长率",
        "近1周",
        "近1月",
        "近3月",
        "近6月",
        "近1年",
        "近2年",
        "近3年",
        "今年来",
        "成立来",
        "成立日期",
        "_",
        "自定义",
        "_",
        "手续费",
        "_",
        "_",
        "_",
        "_",
    ]
    temp_df = temp_df[
        [
            "序号",
            "基金代码",
            "基金简称",
            "拼音缩写",
            "日期",
            "单位净值",
            "累计净值",
            "日增长率",
            "近1周",
            "近1月",
            "近3月",
            "近6月",
            "近1年",
            "近2年",
            "近3年",
            "今年来",
            "成立来",
            "成立日期",
            "自定义",
            "手续费",
        ]
    ]
    temp_df["日期"] = pd.to_datetime(temp_df["日期"], errors="coerce").dt.date
    temp_df["单位净值"] = pd.to_numeric(temp_df["单位净值"], errors="coerce")
    temp_df["累计净值"] = pd.to_numeric(temp_df["累计净值"], errors="coerce")
    temp_df["日增长率"] = pd.to_numeric(temp_df["日增长率"], errors="coerce")
    temp_df["近1周"] = pd.to_numeric(temp_df["近1周"], errors="coerce")
    temp_df["近1月"] = pd.to_numeric(temp_df["近1月"], errors="coerce")
    temp_df["近3月"] = pd.to_numeric(temp_df["近3月"], errors="coerce")
    temp_df["近6月"] = pd.to_numeric(temp_df["近6月"], errors="coerce")
    temp_df["近1年"] = pd.to_numeric(temp_df["近1年"], errors="coerce")
    temp_df["近2年"] = pd.to_numeric(temp_df["近2年"], errors="coerce")
    temp_df["近3年"] = pd.to_numeric(temp_df["近3年"], errors="coerce")
    temp_df["今年来"] = pd.to_numeric(temp_df["今年来"], errors="coerce")
    temp_df["成立来"] = pd.to_numeric(temp_df["成立来"], errors="coerce")
    temp_df["成立日期"] = pd.to_datetime(temp_df["成立日期"], errors="coerce").dt.date
    temp_df["自定义"] = pd.to_numeric(temp_df["自定义"], errors="coerce")
    return temp_df


# Below are copied from https://github.com/quantopian/pyfolio
def get_txn_vol(transactions):
    """
    Extract daily transaction data from set of transaction objects.

    Parameters
    ----------
    transactions : pd.DataFrame
        Time series containing one row per symbol (and potentially
        duplicate datetime indices) and columns for amount and
        price.

    Returns
    -------
    pd.DataFrame
        Daily transaction volume and number of shares.
         - See full explanation in tears.create_full_tear_sheet.
    """

    txn_norm = transactions.copy()
    txn_norm.index = txn_norm.index.normalize()
    amounts = txn_norm.amount.abs()
    prices = txn_norm.price
    values = amounts * prices
    daily_amounts = amounts.groupby(amounts.index).sum()
    daily_values = values.groupby(values.index).sum()
    daily_amounts.name = "txn_shares"
    daily_values.name = "txn_volume"
    return pd.concat([daily_values, daily_amounts], axis=1)


def get_turnover(positions, transactions, denominator="AGB"):
    """
     - Value of purchases and sales divided
    by either the actual gross book or the portfolio value
    for the time step.

    Parameters
    ----------
    positions : pd.DataFrame
        Contains daily position values including cash.
        - See full explanation in tears.create_full_tear_sheet
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    denominator : str, optional
        Either 'AGB' or 'portfolio_value', default AGB.
        - AGB (Actual gross book) is the gross market
        value (GMV) of the specific algo being analyzed.
        Swapping out an entire portfolio of stocks for
        another will yield 200% turnover, not 100%, since
        transactions are being made for both sides.
        - We use average of the previous and the current end-of-period
        AGB to avoid singularities when trading only into or
        out of an entire book in one trading period.
        - portfolio_value is the total value of the algo's
        positions end-of-period, including cash.

    Returns
    -------
    turnover_rate : pd.Series
        timeseries of portfolio turnover rates.
    """

    txn_vol = get_txn_vol(transactions)
    traded_value = txn_vol.txn_volume

    if denominator == "AGB":
        # Actual gross book is the same thing as the algo's GMV
        # We want our denom to be avg(AGB previous, AGB current)
        AGB = positions.drop("cash", axis=1).abs().sum(axis=1)
        denom = AGB.rolling(2).mean()

        # Since the first value of pd.rolling returns NaN, we
        # set our "day 0" AGB to 0.
        denom.iloc[0] = AGB.iloc[0] / 2
    elif denominator == "portfolio_value":
        denom = positions.sum(axis=1)
    else:
        raise ValueError(
            "Unexpected value for denominator '{}'. The "
            "denominator parameter must be either 'AGB'"
            " or 'portfolio_value'.".format(denominator)
        )

    denom.index = denom.index.normalize()
    turnover = traded_value.div(denom, axis="index")
    turnover = turnover.fillna(0)
    return turnover


def gross_lev(positions):
    """
    Calculates the gross leverage of a strategy.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    pd.Series
        Gross leverage.
    """

    exposure = positions.drop("cash", axis=1).abs().sum(axis=1)
    return exposure / positions.sum(axis=1)


def value_at_risk(returns, period=None, sigma=2.0):
    """
    Get value at risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    period : str, optional
        Period over which to calculate VaR. Set to 'weekly',
        'monthly', or 'yearly', otherwise defaults to period of
        returns (typically daily).
    sigma : float, optional
        Standard deviations of VaR, default 2.
    """
    if period is not None:
        returns_agg = ep.aggregate_returns(returns, period)
    else:
        returns_agg = returns.copy()

    value_at_risk = returns_agg.mean() - sigma * returns_agg.std()
    return value_at_risk


SIMPLE_STAT_FUNCS = [
    ep.annual_return,
    ep.cum_returns_final,
    ep.annual_volatility,
    ep.sharpe_ratio,
    ep.calmar_ratio,
    ep.stability_of_timeseries,
    ep.max_drawdown,
    ep.omega_ratio,
    ep.sortino_ratio,
    stats.skew,
    stats.kurtosis,
    ep.tail_ratio,
    value_at_risk,
]


FACTOR_STAT_FUNCS = [
    ep.alpha,
    ep.beta,
]

STAT_FUNC_NAMES = {
    "annual_return": "Annual return",
    "cum_returns_final": "Cumulative returns",
    "annual_volatility": "Annual volatility",
    "sharpe_ratio": "Sharpe ratio",
    "calmar_ratio": "Calmar ratio",
    "stability_of_timeseries": "Stability",
    "max_drawdown": "Max drawdown",
    "omega_ratio": "Omega ratio",
    "sortino_ratio": "Sortino ratio",
    "skew": "Skew",
    "kurtosis": "Kurtosis",
    "tail_ratio": "Tail ratio",
    "common_sense_ratio": "Common sense ratio",
    "value_at_risk": "Daily value at risk",
    "alpha": "Alpha",
    "beta": "Beta",
}


def perf_stats(
    returns,
    factor_returns=None,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
):
    """
    Calculates various performance metrics of a strategy, for use in
    plotting.show_perf_stats.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
         - If None, do not compute alpha, beta, and information ratio.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet.
    turnover_denom : str
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.

    Returns
    -------
    pd.Series
        Performance metrics.
    """

    stats = pd.Series()
    for stat_func in SIMPLE_STAT_FUNCS:
        stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)

    if positions is not None:
        stats["Gross leverage"] = gross_lev(positions).mean()
        if transactions is not None:
            stats["Daily turnover"] = get_turnover(
                positions, transactions, turnover_denom
            ).mean()
    if factor_returns is not None:
        for stat_func in FACTOR_STAT_FUNCS:
            res = stat_func(returns, factor_returns)
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = res

    return stats


STAT_FUNCS_PCT = [
    "Annual return",
    "Cumulative returns",
    "Annual volatility",
    "Max drawdown",
    "Daily value at risk",
    "Daily turnover",
]


def plotting_context(context="notebook", font_scale=1.5, rc=None):
    """
    Create pyfolio default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    >>> with pyfolio.plotting.plotting_context(font_scale=2):
    >>>    pyfolio.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {"lines.linewidth": 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return seaborn.plotting_context(context=context, font_scale=font_scale, rc=rc)


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.
    """

    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop("set_context", True)
        if set_context:
            with plotting_context(), axes_style():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return call_w_context


def axes_style(style="darkgrid", rc=None):
    """
    Create pyfolio default axes style context.

    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    >>> with pyfolio.plotting.axes_style(style='whitegrid'):
    >>>    pyfolio.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return seaborn.axes_style(style=style, rc=rc)


def clip_returns_to_benchmark(rets, benchmark_rets):
    """
    Drop entries from rets so that the start and end dates of rets match those
    of benchmark_rets.

    Parameters
    ----------
    rets : pd.Series
        Daily returns of the strategy, noncumulative.
         - See pf.tears.create_full_tear_sheet for more details

    benchmark_rets : pd.Series
        Daily returns of the benchmark, noncumulative.

    Returns
    -------
    clipped_rets : pd.Series
        Daily noncumulative returns with index clipped to match that of
        benchmark returns.
    """

    if (rets.index[0] < benchmark_rets.index[0]) or (
        rets.index[-1] > benchmark_rets.index[-1]
    ):
        clipped_rets = rets[benchmark_rets.index]
    else:
        clipped_rets = rets

    return clipped_rets


def print_table(table, name=None, float_format=None, formatters=None, header_rows=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pandas.Series or pandas.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    float_format : function, optional
        Formatter to use for displaying table elements, passed as the
        `float_format` arg to pd.Dataframe.to_html.
        E.g. `'{0:.2%}'.format` for displaying 100 as '100.00%'.
    formatters : list or dict, optional
        Formatters to use by column, passed as the `formatters` arg to
        pd.Dataframe.to_html.
    header_rows : dict, optional
        Extra rows to display at the top of the table.
    """

    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if name is not None:
        table.columns.name = name

    html = table.to_html(float_format=float_format, formatters=formatters)

    if header_rows is not None:
        # Count the number of columns for the text to span
        n_cols = html.split("<thead>")[1].split("</thead>")[0].count("<th>")

        # Generate the HTML for the extra rows
        rows = ""
        for name, value in header_rows.items():
            rows += (
                '\n    <tr style="text-align: right;"><th>%s</th>'
                + "<td colspan=%d>%s</td></tr>"
            ) % (name, n_cols, value)

        # Inject the new HTML
        html = html.replace("<thead>", "<thead>" + rows)

    display(HTML(html))


def show_perf_stats(
    returns,
    factor_returns=None,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
    live_start_date=None,
    bootstrap=False,
    header_rows=None,
):
    """
    Prints some performance metrics of the strategy.

    - Shows amount of time the strategy has been run in backtest and
      out-of-sample (in live trading).

    - Shows Omega ratio, max drawdown, Calmar ratio, annual return,
      stability, Sharpe ratio, annual volatility, alpha, and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics.
         - For more information, see timeseries.perf_stats_bootstrap
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the displayed table.
    """

    perf_stats_all = perf_stats(
        returns,
        factor_returns=factor_returns,
        positions=positions,
        transactions=transactions,
        turnover_denom=turnover_denom,
    )

    date_rows = OrderedDict()
    if len(returns.index) > 0:
        date_rows["Start date"] = returns.index[0].strftime("%Y-%m-%d")
        date_rows["End date"] = returns.index[-1].strftime("%Y-%m-%d")
        date_rows["Total months"] = int(len(returns) / config.APPROX_BDAYS_PER_MONTH)
    perf_stats_var = pd.DataFrame(perf_stats_all, columns=["Backtest"])
    perf_stats_str = perf_stats_var.astype("str")

    for column in perf_stats_var.columns:
        for stat, value in perf_stats_var[column].items():
            if stat in STAT_FUNCS_PCT:
                perf_stats_str.loc[stat, column] = str(np.round(value * 100, 3)) + "%"
    perf_stats_var = perf_stats_str
    if header_rows is None:
        header_rows = date_rows
    else:
        header_rows = OrderedDict(header_rows)
        header_rows.update(date_rows)

    print_table(
        perf_stats_var,
        float_format="{0:.2f}".format,
        header_rows=header_rows,
    )


def get_max_drawdown_underwater(underwater):
    """
    Determines peak, valley, and recovery dates given an 'underwater'
    DataFrame.

    An underwater DataFrame is a DataFrame that has precomputed
    rolling drawdown.

    Parameters
    ----------
    underwater : pd.Series
       Underwater returns (rolling drawdown) of a strategy.

    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.
    """

    valley = underwater.idxmin()  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery


def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """

    returns = returns.copy()
    df_cum = ep.cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak:recovery].index[1:-1], inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if (len(returns) == 0) or (len(underwater) == 0) or (np.min(underwater) == 0):
            break

    return drawdowns


def gen_drawdown_table(returns, top=10):
    """
    Places top drawdowns in a table.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """

    df_cum = ep.cum_returns(returns, 1.0)
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(
        index=list(range(top)),
        columns=[
            "Net drawdown in %",
            "Peak date",
            "Valley date",
            "Recovery date",
            "Duration",
        ],
    )

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, "Duration"] = np.nan
        else:
            df_drawdowns.loc[i, "Duration"] = len(
                pd.date_range(peak, recovery, freq="B")
            )
        df_drawdowns.loc[i, "Peak date"] = peak.to_pydatetime().strftime("%Y-%m-%d")
        df_drawdowns.loc[i, "Valley date"] = valley.to_pydatetime().strftime("%Y-%m-%d")
        if isinstance(recovery, float):
            df_drawdowns.loc[i, "Recovery date"] = recovery
        else:
            df_drawdowns.loc[i, "Recovery date"] = recovery.to_pydatetime().strftime(
                "%Y-%m-%d"
            )
        df_drawdowns.loc[i, "Net drawdown in %"] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]
        ) * 100

    df_drawdowns["Peak date"] = pd.to_datetime(df_drawdowns["Peak date"])
    df_drawdowns["Valley date"] = pd.to_datetime(df_drawdowns["Valley date"])
    df_drawdowns["Recovery date"] = pd.to_datetime(df_drawdowns["Recovery date"])

    return df_drawdowns


def show_worst_drawdown_periods(returns, top=5):
    """
    Prints information about the worst drawdown periods.

    Prints peak dates, valley dates, recovery dates, and net
    drawdowns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 5).
    """

    drawdown_df = gen_drawdown_table(returns, top=top)
    print_table(
        drawdown_df.sort_values("Net drawdown in %", ascending=False),
        name="Worst drawdown periods",
        float_format="{0:.2f}".format,
    )


def simulate_paths(
    is_returns, num_days, starting_value=1, num_samples=1000, random_seed=None
):
    """
    Gnerate alternate paths using available values from in-sample returns.

    Parameters
    ----------
    is_returns : pandas.core.frame.DataFrame
        Non-cumulative in-sample returns.
    num_days : int
        Number of days to project the probability cone forward.
    starting_value : int or float
        Starting value of the out of sample period.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.

    Returns
    -------
    samples : numpy.ndarray
    """

    samples = np.empty((num_samples, num_days))
    seed = np.random.RandomState(seed=random_seed)
    for i in range(num_samples):
        samples[i, :] = is_returns.sample(num_days, replace=True, random_state=seed)

    return samples


def summarize_paths(samples, cone_std=(1.0, 1.5, 2.0), starting_value=1.0):
    """
    Gnerate the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns.

    Parameters
    ----------
    samples : numpy.ndarray
        Alternative paths, or series of possible outcomes.
    cone_std : list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.

    Returns
    -------
    samples : pandas.core.frame.DataFrame
    """

    cum_samples = ep.cum_returns(samples.T, starting_value=starting_value).T

    cum_mean = cum_samples.mean(axis=0)
    cum_std = cum_samples.std(axis=0)

    if isinstance(cone_std, (float, int)):
        cone_std = [cone_std]

    cone_bounds = pd.DataFrame(columns=pd.Float64Index([]))
    for num_std in cone_std:
        cone_bounds.loc[:, float(num_std)] = cum_mean + cum_std * num_std
        cone_bounds.loc[:, float(-num_std)] = cum_mean - cum_std * num_std

    return cone_bounds


def forecast_cone_bootstrap(
    is_returns,
    num_days,
    cone_std=(1.0, 1.5, 2.0),
    starting_value=1,
    num_samples=1000,
    random_seed=None,
):
    """
    Determines the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns. Future cumulative mean and
    standard devation are computed by repeatedly sampling from the
    in-sample daily returns (i.e. bootstrap). This cone is non-parametric,
    meaning it does not assume that returns are normally distributed.

    Parameters
    ----------
    is_returns : pd.Series
        In-sample daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    num_days : int
        Number of days to project the probability cone forward.
    cone_std : int, float, or list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    starting_value : int or float
        Starting value of the out of sample period.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.

    Returns
    -------
    pd.DataFrame
        Contains upper and lower cone boundaries. Column names are
        strings corresponding to the number of standard devations
        above (positive) or below (negative) the projected mean
        cumulative returns.
    """

    samples = simulate_paths(
        is_returns=is_returns,
        num_days=num_days,
        starting_value=starting_value,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    cone_bounds = summarize_paths(
        samples=samples, cone_std=cone_std, starting_value=starting_value
    )

    return cone_bounds


def two_dec_places(x, pos):
    """
    Adds 1/100th decimal to plot ticks.
    """

    return "%.2f" % x


def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """

    return "%.0f%%" % x


def plot_rolling_returns(
    returns,
    factor_returns=None,
    live_start_date=None,
    logy=False,
    cone_std=None,
    legend_loc="best",
    volatility_match=False,
    cone_function=forecast_cone_bootstrap,
    ax=None,
    **kwargs
):
    """
    Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    logy : bool, optional
        Whether to log-scale the y-axis.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    volatility_match : bool, optional
        Whether to normalize the volatility of the returns to those of the
        benchmark returns. This helps compare strategies with different
        volatilities. Requires passing of benchmark_rets.
    cone_function : function, optional
        Function to use when generating forecast probability cone.
        The function signiture must follow the form:
        def cone(in_sample_returns (pd.Series),
                 days_to_project_forward (int),
                 cone_std= (float, or tuple),
                 starting_value= (int, or float))
        See timeseries.forecast_cone_bootstrap for an example.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel("")
    ax.set_ylabel("Cumulative returns")
    ax.set_yscale("log" if logy else "linear")

    if volatility_match and factor_returns is None:
        raise ValueError("volatility_match requires passing of " "factor_returns.")
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = ep.cum_returns(returns, 1.0)

    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        cum_factor_returns = ep.cum_returns(factor_returns[cum_rets.index], 1.0)
        cum_factor_returns.plot(
            lw=2, color="gray", label=factor_returns.name, alpha=0.60, ax=ax, **kwargs
        )

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    is_cum_returns.plot(
        lw=3, color="forestgreen", alpha=0.6, label="Backtest", ax=ax, **kwargs
    )

    if len(oos_cum_returns) > 0:
        oos_cum_returns.plot(
            lw=4, color="red", alpha=0.6, label="Live", ax=ax, **kwargs
        )

        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function(
                is_returns,
                len(oos_cum_returns),
                cone_std=cone_std,
                starting_value=is_cum_returns[-1],
            )

            cone_bounds = cone_bounds.set_index(oos_cum_returns.index)
            for std in cone_std:
                ax.fill_between(
                    cone_bounds.index,
                    cone_bounds[float(std)],
                    cone_bounds[float(-std)],
                    color="steelblue",
                    alpha=0.5,
                )

    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.5)
    ax.axhline(1.0, linestyle="--", color="black", lw=2)

    return ax


def plot_returns(returns, live_start_date=None, ax=None):
    """
    Plots raw returns over time.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_label("")
    ax.set_ylabel("Returns")

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_returns = returns.loc[returns.index < live_start_date]
        oos_returns = returns.loc[returns.index >= live_start_date]
        is_returns.plot(ax=ax, color="g")
        oos_returns.plot(ax=ax, color="r")

    else:
        returns.plot(ax=ax, color="g")

    return ax


def rolling_beta(
    returns, factor_returns, rolling_window=config.APPROX_BDAYS_PER_MONTH * 6
):
    """
    Determines the rolling beta of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series or pd.DataFrame
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - If DataFrame is passed, computes rolling beta for each column.
         - This is in the same style as returns.
    rolling_window : int, optional
        The size of the rolling window, in days, over which to compute
        beta (default 6 months).

    Returns
    -------
    pd.Series
        Rolling beta.

    Note
    -----
    See https://en.wikipedia.org/wiki/Beta_(finance) for more details.
    """

    if factor_returns.ndim > 1:
        # Apply column-wise
        return factor_returns.apply(
            partial(rolling_beta, returns), rolling_window=rolling_window
        )
    else:
        out = pd.Series(index=returns.index)
        for beg, end in zip(
            returns.index[0:-rolling_window], returns.index[rolling_window:]
        ):
            out.loc[end] = ep.beta(returns.loc[beg:end], factor_returns.loc[beg:end])

        return out


def rolling_volatility(returns, rolling_vol_window):
    """
    Determines the rolling volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_vol_window : int
        Length of rolling window, in days, over which to compute.

    Returns
    -------
    pd.Series
        Rolling volatility.
    """

    return returns.rolling(rolling_vol_window).std() * np.sqrt(
        config.APPROX_BDAYS_PER_YEAR
    )


def rolling_sharpe(returns, rolling_sharpe_window):
    """
    Determines the rolling Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_sharpe_window : int
        Length of rolling window, in days, over which to compute.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """

    return (
        returns.rolling(rolling_sharpe_window).mean()
        / returns.rolling(rolling_sharpe_window).std()
        * np.sqrt(config.APPROX_BDAYS_PER_YEAR)
    )


def plot_rolling_beta(returns, factor_returns, legend_loc="best", ax=None, **kwargs):
    """
    Plots the rolling 6-month and 12-month beta versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title("Rolling portfolio beta to " + str(factor_returns.name))
    ax.set_ylabel("Beta")
    rb_1 = rolling_beta(
        returns, factor_returns, rolling_window=config.APPROX_BDAYS_PER_MONTH * 6
    )
    rb_1.plot(color="steelblue", lw=3, alpha=0.6, ax=ax, **kwargs)
    rb_2 = rolling_beta(
        returns, factor_returns, rolling_window=config.APPROX_BDAYS_PER_MONTH * 12
    )
    rb_2.plot(color="grey", lw=3, alpha=0.4, ax=ax, **kwargs)
    ax.axhline(rb_1.mean(), color="steelblue", linestyle="--", lw=3)
    ax.axhline(0.0, color="black", linestyle="-", lw=2)

    ax.set_xlabel("")
    ax.legend(["6-mo", "12-mo"], loc=legend_loc, frameon=True, framealpha=0.5)
    ax.set_ylim((-1.0, 1.0))
    return ax


def plot_rolling_volatility(
    returns,
    factor_returns=None,
    rolling_window=config.APPROX_BDAYS_PER_MONTH * 6,
    legend_loc="best",
    ax=None,
    **kwargs
):
    """
    Plots the rolling volatility versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for which the
        benchmark rolling volatility is computed. Usually a benchmark such
        as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the volatility.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_vol_ts = rolling_volatility(returns, rolling_window)
    rolling_vol_ts.plot(alpha=0.7, lw=3, color="orangered", ax=ax, **kwargs)
    if factor_returns is not None:
        rolling_vol_ts_factor = rolling_volatility(factor_returns, rolling_window)
        rolling_vol_ts_factor.plot(alpha=0.7, lw=3, color="grey", ax=ax, **kwargs)

    ax.set_title("Rolling volatility (6-month)")
    ax.axhline(rolling_vol_ts.mean(), color="steelblue", linestyle="--", lw=3)

    ax.axhline(0.0, color="black", linestyle="-", lw=2)

    ax.set_ylabel("Volatility")
    ax.set_xlabel("")
    if factor_returns is None:
        ax.legend(
            ["Volatility", "Average volatility"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )
    else:
        ax.legend(
            ["Volatility", "Benchmark volatility", "Average volatility"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )
    return ax


def plot_rolling_sharpe(
    returns,
    factor_returns=None,
    rolling_window=config.APPROX_BDAYS_PER_MONTH * 6,
    legend_loc="best",
    ax=None,
    **kwargs
):
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for
        which the benchmark rolling Sharpe is computed. Usually
        a benchmark such as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the sharpe ratio.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_sharpe_ts = rolling_sharpe(returns, rolling_window)
    rolling_sharpe_ts.plot(alpha=0.7, lw=3, color="orangered", ax=ax, **kwargs)

    if factor_returns is not None:
        rolling_sharpe_ts_factor = rolling_sharpe(factor_returns, rolling_window)
        rolling_sharpe_ts_factor.plot(alpha=0.7, lw=3, color="grey", ax=ax, **kwargs)

    ax.set_title("Rolling Sharpe ratio (6-month)")
    ax.axhline(rolling_sharpe_ts.mean(), color="steelblue", linestyle="--", lw=3)
    ax.axhline(0.0, color="black", linestyle="-", lw=3)

    ax.set_ylabel("Sharpe ratio")
    ax.set_xlabel("")
    if factor_returns is None:
        ax.legend(["Sharpe", "Average"], loc=legend_loc, frameon=True, framealpha=0.5)
    else:
        ax.legend(
            ["Sharpe", "Benchmark Sharpe", "Average"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )

    return ax


def plot_drawdown_periods(returns, top=10, ax=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    df_drawdowns = gen_drawdown_table(returns, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = seaborn.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[["Peak date", "Recovery date"]].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between((peak, recovery), lim[0], lim[1], alpha=0.4, color=colors[i])
    ax.set_ylim(lim)
    ax.set_title("Top %i drawdown periods" % top)
    ax.set_ylabel("Cumulative returns")
    ax.legend(["Portfolio"], loc="upper left", frameon=True, framealpha=0.5)
    ax.set_xlabel("")
    return ax


def plot_drawdown_underwater(returns, ax=None, **kwargs):
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    (underwater).plot(ax=ax, kind="area", color="coral", alpha=0.7, **kwargs)
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater plot")
    ax.set_xlabel("")
    return ax


def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
    """
    Plots a heatmap of returns by month.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    monthly_ret_table = ep.aggregate_returns(returns, "monthly")
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    seaborn.heatmap(
        monthly_ret_table.fillna(0) * 100.0,
        annot=True,
        annot_kws={"size": 9},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=matplotlib.cm.RdYlGn,
        ax=ax,
        **kwargs
    )
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    ax.set_title("Monthly returns (%)")
    return ax


def plot_annual_returns(returns, ax=None, **kwargs):
    """
    Plots a bar graph of returns by year.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis="x", which="major")

    ann_ret_df = pd.DataFrame(ep.aggregate_returns(returns, "yearly"))

    ax.axvline(
        100 * ann_ret_df.values.mean(),
        color="steelblue",
        linestyle="--",
        lw=4,
        alpha=0.7,
    )
    (100 * ann_ret_df.sort_index(ascending=False)).plot(
        ax=ax, kind="barh", alpha=0.70, **kwargs
    )
    ax.axvline(0.0, color="black", linestyle="-", lw=3)

    ax.set_ylabel("Year")
    ax.set_xlabel("Returns")
    ax.set_title("Annual returns")
    ax.legend(["Mean"], frameon=True, framealpha=0.5)
    return ax


def plot_monthly_returns_dist(returns, ax=None, **kwargs):
    """
    Plots a distribution of monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis="x", which="major")

    monthly_ret_table = ep.aggregate_returns(returns, "monthly")

    ax.hist(100 * monthly_ret_table, color="orangered", alpha=0.80, bins=20, **kwargs)

    ax.axvline(
        100 * monthly_ret_table.mean(), color="gold", linestyle="--", lw=4, alpha=1.0
    )

    ax.axvline(0.0, color="black", linestyle="-", lw=3, alpha=0.75)
    ax.legend(["Mean"], frameon=True, framealpha=0.5)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Returns")
    ax.set_title("Distribution of monthly returns")
    return ax


@customize
def create_returns_tear_sheet(
    returns,
    positions=None,
    transactions=None,
    live_start_date=None,
    cone_std=(1.0, 1.5, 2.0),
    benchmark_rets=None,
    bootstrap=False,
    turnover_denom="AGB",
    header_rows=None,
    return_fig=False,
):
    """
    Generate a number of plots for analyzing a strategy's returns.

    - Fetches benchmarks, then creates the plots on a single figure.
    - Plots: rolling returns (with cone), rolling beta, rolling sharpe,
        rolling Fama-French risk factors, drawdowns, underwater plot, monthly
        and annual return plots, daily similarity plots,
        and return quantile box plot.
    - Will also print the start and end dates of the strategy,
        performance statistics, drawdown periods, and the return range.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
        - See full explanation in create_full_tear_sheet.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - The cone is a normal distribution with this standard deviation
             centered around a linear regression.
    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics. Takes a few minutes longer.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    if benchmark_rets is not None:
        returns = clip_returns_to_benchmark(returns, benchmark_rets)

    show_perf_stats(
        returns,
        benchmark_rets,
        positions=positions,
        transactions=transactions,
        turnover_denom=turnover_denom,
        bootstrap=bootstrap,
        live_start_date=live_start_date,
        header_rows=header_rows,
    )

    show_worst_drawdown_periods(returns)

    vertical_sections = 11

    if live_start_date is not None:
        vertical_sections += 1
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    if benchmark_rets is not None:
        vertical_sections += 1

    if bootstrap:
        vertical_sections += 1

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_rolling_returns = plt.subplot(gs[:2, :])

    i = 2
    ax_rolling_returns_vol_match = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_rolling_returns_log = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_returns = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    if benchmark_rets is not None:
        ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
    ax_rolling_volatility = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_drawdown = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_underwater = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_monthly_heatmap = plt.subplot(gs[i, 0])
    ax_annual_returns = plt.subplot(gs[i, 1])
    ax_monthly_dist = plt.subplot(gs[i, 2])
    i += 1

    plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns,
    )
    ax_rolling_returns.set_title("Cumulative returns")

    plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=None,
        volatility_match=(benchmark_rets is not None),
        legend_loc=None,
        ax=ax_rolling_returns_vol_match,
    )
    ax_rolling_returns_vol_match.set_title(
        "Cumulative returns volatility matched to benchmark"
    )

    plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        logy=True,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns_log,
    )
    ax_rolling_returns_log.set_title("Cumulative returns on logarithmic scale")

    plot_returns(
        returns,
        live_start_date=live_start_date,
        ax=ax_returns,
    )
    ax_returns.set_title("Returns")

    if benchmark_rets is not None:
        plot_rolling_beta(returns, benchmark_rets, ax=ax_rolling_beta)

    plot_rolling_volatility(
        returns, factor_returns=benchmark_rets, ax=ax_rolling_volatility
    )

    plot_rolling_sharpe(returns, ax=ax_rolling_sharpe)

    # Drawdowns
    plot_drawdown_periods(returns, top=5, ax=ax_drawdown)

    plot_drawdown_underwater(returns=returns, ax=ax_underwater)

    plot_monthly_returns_heatmap(returns, ax=ax_monthly_heatmap)
    plot_annual_returns(returns, ax=ax_annual_returns)
    plot_monthly_returns_dist(returns, ax=ax_monthly_dist)

    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)

    if return_fig:
        return fig
