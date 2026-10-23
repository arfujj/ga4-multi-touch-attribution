-- One row per touch (session) in a user journey.
--
-- A journey is the ordered sequence of a user's sessions that ends in a purchase session
-- (converted) or at the end of the data (not converted). Sessions after a purchase start a
-- new journey. Only touches within {{ var('lookback_days') }} days before the journey's end
-- (purchase time, or the last session for non-converting journeys) are kept.

with sessions as (

    select
        session_key,
        user_pseudo_id,
        session_start_ts,
        channel,
        converted,
        first_purchase_ts,
        revenue_usd
    from {{ ref('fct_sessions') }}

),

numbered as (

    select
        *,
        -- journeys index = number of converting sessions strictly before this one
        coalesce(sum(if(converted, 1, 0)) over (
            partition by user_pseudo_id
            order by session_start_ts, session_key
            rows between unbounded preceding and 1 preceding
        ), 0) as journey_index
    from sessions

),

journey_end as (

    select
        user_pseudo_id,
        journey_index,
        logical_or(converted) as journey_converted,
        max(if(converted, first_purchase_ts, null)) as conversion_ts,
        max(session_start_ts) as last_touch_ts
    from numbered
    group by 1, 2

),

windowed as (

    select
        concat(n.user_pseudo_id, '#', cast(n.journey_index as string)) as journey_id,
        n.user_pseudo_id,
        n.session_key,
        n.session_start_ts as touch_ts,
        n.channel,
        n.converted as is_conversion_touch,
        n.revenue_usd,
        e.journey_converted,
        e.conversion_ts,
        coalesce(e.conversion_ts, e.last_touch_ts) as journey_end_ts
    from numbered as n
    inner join journey_end as e using (user_pseudo_id, journey_index)
    where n.session_start_ts >= timestamp_sub(
        coalesce(e.conversion_ts, e.last_touch_ts), interval {{ var('lookback_days') }} day
    )

)

select
    journey_id,
    user_pseudo_id,
    row_number() over (partition by journey_id order by touch_ts, session_key) as touch_number,
    session_key,
    touch_ts,
    channel,
    is_conversion_touch,
    revenue_usd,
    journey_converted,
    conversion_ts,
    journey_end_ts
from windowed
