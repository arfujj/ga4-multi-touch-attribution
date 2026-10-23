-- One row per session with its channel group.

select
    s.*,
    m.channel_group as channel,
    m.rule_name as channel_rule
from {{ ref('int_sessions') }} as s
left join {{ ref('int_channel_map') }} as m
    on lower(coalesce(s.source, '(null)')) = m.source_key
    and lower(coalesce(s.medium, '(null)')) = m.medium_key
