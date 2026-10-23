-- Fails if purchase counts disagree between raw events (de-duplicated), fct_purchases,
-- fct_sessions, and the journey layer (every converting session ends exactly one journey).

with raw as (

    select count(*) as n
    from (
      select * from (
        select
            user_pseudo_id,
            (select value.int_value from unnest(event_params) where key = 'ga_session_id') as sid,
            event_timestamp,
            coalesce(
            ecommerce.transaction_id,
            (select value.string_value from unnest(event_params) where key = 'transaction_id'),
            cast((select value.int_value from unnest(event_params) where key = 'transaction_id') as string)
        ) as tid
        from {{ source('ga4', 'events') }}
        where _TABLE_SUFFIX between '{{ var("start_suffix") }}' and '{{ var("end_suffix") }}'
          and event_name = 'purchase'
      )
      where true
      qualify not (
          tid is not null and tid != '(not set)'
          and row_number() over (partition by user_pseudo_id, sid, tid order by event_timestamp) > 1
      )
    )

),

checks as (

    select 'fct_purchases rows vs raw' as check_name,
        (select count(*) from {{ ref('fct_purchases') }}) as modeled, (select n from raw) as expected
    union all
    select 'sum(fct_sessions.n_purchases) vs raw',
        (select coalesce(sum(n_purchases), 0) from {{ ref('fct_sessions') }}), (select n from raw)
    union all
    select 'converted journeys vs converted sessions',
        (select countif(converted) from {{ ref('fct_journeys') }}),
        (select countif(converted) from {{ ref('fct_sessions') }})
    union all
    select 'sessions in fct_sessions vs distinct sessions in staging',
        (select count(*) from {{ ref('fct_sessions') }}),
        (select count(distinct session_key) from {{ ref('stg_ga4__events') }})

)

select * from checks
where modeled != expected
   or expected = 0   -- an empty pipeline must not pass
