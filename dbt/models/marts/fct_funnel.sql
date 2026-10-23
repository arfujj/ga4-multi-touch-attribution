-- Closed session funnel by channel and device: a session counts at step k only if it
-- also reached every earlier step. Counts are additive across channel/device.

with s as (

    select
        channel,
        device_category,
        true as s1,
        has_view_item as s2,
        has_view_item and has_add_to_cart as s3,
        has_view_item and has_add_to_cart and has_begin_checkout as s4,
        has_view_item and has_add_to_cart and has_begin_checkout and has_add_payment_info as s5,
        has_view_item and has_add_to_cart and has_begin_checkout and has_add_payment_info and converted as s6,
        has_view_item as o2,
        has_add_to_cart as o3,
        has_begin_checkout as o4,
        has_add_payment_info as o5,
        converted as o6
    from {{ ref('fct_sessions') }}

),

steps as (

    select channel, device_category, step_order, step, reached_closed, reached_open
    from s,
    unnest([
        struct(1 as step_order, 'session_start' as step, s1 as reached_closed, s1 as reached_open),
        (2, 'view_item', s2, o2),
        (3, 'add_to_cart', s3, o3),
        (4, 'begin_checkout', s4, o4),
        (5, 'add_payment_info', s5, o5),
        (6, 'purchase', s6, o6)
    ])

)

select
    channel,
    device_category,
    step_order,
    step,
    countif(reached_closed) as sessions_closed,
    countif(reached_open) as sessions_open
from steps
group by 1, 2, 3, 4
