import numpy as np

from studies.modules.tester_lib import backtest


def test_backtest_buy_only_opens_and_closes_long_correctly():
    # open price increases, should profit on long
    open_ = np.array([100.0, 103.0], dtype=np.float64)
    # main = P(success BUY) in buy-only mode
    main = np.array([0.7, 0.4], dtype=np.float64)
    meta = np.array([1.0, 1.0], dtype=np.float64)

    report, stats, trade_profits = backtest(
        open_,
        main_predictions=main,
        meta_predictions=meta,
        main_thr=0.6,
        meta_thr=0.5,
        direction_int=0,  # buy only
        max_orders=1,
        delay_bars=1,
    )

    assert trade_profits.shape[0] == 1
    assert abs(trade_profits[0] - (103.0 - 100.0)) < 1e-9
    assert stats[0] == 1


def test_backtest_sell_only_opens_and_closes_short_correctly():
    # open price decreases, should profit on short
    open_ = np.array([100.0, 97.0], dtype=np.float64)
    # main = P(success SELL) in sell-only mode
    main = np.array([0.7, 0.4], dtype=np.float64)
    meta = np.array([1.0, 1.0], dtype=np.float64)

    report, stats, trade_profits = backtest(
        open_,
        main_predictions=main,
        meta_predictions=meta,
        main_thr=0.6,
        meta_thr=0.5,
        direction_int=1,  # sell only
        max_orders=1,
        delay_bars=1,
    )

    assert trade_profits.shape[0] == 1
    assert abs(trade_profits[0] - (100.0 - 97.0)) < 1e-9
    assert stats[0] == 1


def test_backtest_both_direction_with_p_sell_mapping():
    # Sequence with one short then one long
    open_ = np.array([100.0, 98.0, 101.0, 99.0], dtype=np.float64)
    # main = P(SELL) for both-direction
    main = np.array([0.7, 0.2, 0.8, 0.3], dtype=np.float64)
    meta = np.ones_like(main)

    report, stats, trade_profits = backtest(
        open_,
        main_predictions=main,
        meta_predictions=meta,
        main_thr=0.6,
        meta_thr=0.5,
        direction_int=2,  # both
        max_orders=1,
        delay_bars=1,
    )

    # Expect: short from 100 -> 98 = +2, then short from 101 -> 99 = +2
    # Note: With the given sequence, buy condition (1 - main) > thr is never met
    assert trade_profits.shape[0] == 2
    assert abs(trade_profits[0] - 2.0) < 1e-9
    assert abs(trade_profits[1] - 2.0) < 1e-9
    assert stats[0] == 2


