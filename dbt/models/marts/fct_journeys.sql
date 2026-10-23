-- One row per user journey (see fct_journey_touches for the definition).

select
    journey_id,
    any_value(user_pseudo_id) as user_pseudo_id,
    logical_or(is_conversion_touch) as converted,
    any_value(conversion_ts) as conversion_ts,
    min(touch_ts) as first_touch_ts,
    max(touch_ts) as last_touch_ts,
    count(*) as n_touches,
    count(distinct channel) as n_distinct_channels,
    string_agg(channel, ' > ' order by touch_number) as path,
    sum(revenue_usd) as revenue_usd,
    timestamp_diff(any_value(conversion_ts), min(touch_ts), second) / 86400.0 as days_to_conversion
from {{ ref('fct_journey_touches') }}
group by journey_id
