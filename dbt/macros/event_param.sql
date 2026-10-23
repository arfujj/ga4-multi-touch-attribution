{# Extract one key from the event_params array. value_type: string | int | double | float #}
{% macro event_param(key, value_type='string') -%}
(SELECT value.{{ value_type }}_value FROM UNNEST(event_params) WHERE key = '{{ key }}')
{%- endmacro %}
