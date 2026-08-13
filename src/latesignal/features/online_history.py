"""Past-only cold status and frequency state."""

from __future__ import annotations

from collections import Counter


class OnlineHistory:
    def __init__(self) -> None:
        self._users: Counter[str] = Counter()
        self._products: Counter[str] = Counter()

    def observe(self, user_id: str, product_id: str) -> dict[str, int | bool]:
        prior_user = self._users[user_id]
        prior_product = self._products[product_id]
        result: dict[str, int | bool] = {
            "cold_user": prior_user == 0,
            "cold_product": prior_product == 0,
            "prior_user_clicks": prior_user,
            "prior_product_clicks": prior_product,
        }
        self._users[user_id] += 1
        self._products[product_id] += 1
        return result
