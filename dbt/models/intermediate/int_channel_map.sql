-- One row per observed source/medium combination, mapped to a channel group by the
-- first matching rule (lowest priority number) in the channel_rules seed.

with combos as (

    select
        lower(coalesce(source, '(null)')) as source_key,
        lower(coalesce(medium, '(null)')) as medium_key,
        count(*) as sessions
    from {{ ref('int_sessions') }}
    group by 1, 2

),

matched as (

    select
        c.source_key,
        c.medium_key,
        c.sessions,
        r.channel_group,
        r.rule_name,
        r.priority,
        row_number() over (partition by c.source_key, c.medium_key order by r.priority) as rn
    from combos as c
    inner join {{ ref('channel_rules') }} as r
        on regexp_contains(c.source_key, r.source_regex)
        and regexp_contains(c.medium_key, r.medium_regex)

)

select
    concat(source_key, ' / ', medium_key) as source_medium_key,
    source_key,
    medium_key,
    sessions,
    channel_group,
    rule_name
from matched
where rn = 1
