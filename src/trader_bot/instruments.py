"""Instrument metadata and pip-value maths.

Handles the two things that trip up FX position sizing:
  * JPY pairs use a 0.01 pip size; everything else uses 0.0001.
  * pip value in the account currency depends on the quote currency.

We assume a USD account (see config). For XXX/USD pairs the pip value per unit
is a constant 0.0001 USD. For USD/JPY the pip value per unit is 0.01 / price USD
(price expressed in JPY per USD).
"""
from __future__ import annotations


def pip_size(pair: str) -> float:
    """0.01 for JPY-quoted pairs, else 0.0001."""
    quote = pair.split("_")[1]
    return 0.01 if quote == "JPY" else 0.0001


def pip_value_per_unit(pair: str, price: float, account_ccy: str = "USD") -> float:
    """Value of a 1-pip move for ONE unit of the base currency, in account ccy.

    Parameters
    ----------
    pair : e.g. "EUR_USD" or "USD_JPY"
    price : current price of the pair (quote per base)
    """
    base, quote = pair.split("_")
    ps = pip_size(pair)

    if quote == account_ccy:
        # e.g. EUR/USD with USD account: 1 unit moves `ps` USD per pip.
        return ps
    if base == account_ccy:
        # e.g. USD/JPY with USD account: pip value = ps / price (JPY -> USD).
        return ps / price
    # Cross pair without the account ccy — would need a conversion rate.
    # Not used in the v1 universe (all pairs contain USD).
    raise ValueError(
        f"pip_value_per_unit: pair {pair} does not contain account ccy "
        f"{account_ccy}; a cross-rate is required."
    )


def price_to_pips(pair: str, price_distance: float) -> float:
    """Convert an absolute price distance to pips."""
    return price_distance / pip_size(pair)


def pips_to_price(pair: str, pips: float) -> float:
    """Convert pips to an absolute price distance."""
    return pips * pip_size(pair)
