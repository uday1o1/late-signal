"""Reviewed raw schema roles used by the data audit."""

OUTCOME_FIELDS = frozenset({"Sale", "SalesAmountInEuro"})
TIMING_FIELDS = frozenset({"time_delay_for_conversion", "click_timestamp"})
NUMERIC_CLICK_FIELDS = frozenset({"nb_clicks_1week", "product_price"})
CATEGORICAL_CLICK_FIELDS = frozenset(
    {
        "product_age_group",
        "device_type",
        "audience_id",
        "product_gender",
        "product_brand",
        "product_category1",
        "product_category2",
        "product_category3",
        "product_category4",
        "product_category5",
        "product_category6",
        "product_category7",
        "product_country",
        "product_id",
        "product_title",
        "partner_id",
        "user_id",
    }
)

RAW_FIELD_COUNT = 23
