-- Fails if any session ends before it starts.
select session_key, session_duration_sec
from {{ ref('fct_sessions') }}
where session_duration_sec < 0
