{% extends 'markdown.tpl' %}

<!-- Add Div for input area -->
{% block input %}
<div class="input_area" markdown="1">
{{ super() }}
</div>
{% endblock input %}

<!-- Remove indentations for output text and add div classes  -->
{% block stream %}
<div class="output_area" markdown="1">
{{ super() }}
</div>
{% endblock stream %}


{% block data_text %}
<div class="output_area" markdown="1">
{{ super() }}
</div>
{% endblock data_text %}


{% block traceback_line  %}
<div class="output_area" markdown="1">
{{ super() }}
</div>
{% endblock traceback_line  %}

<!-- Tell Jekyll not to render HTML output blocks as markdown -->
{% block data_html %}
<div markdown="0">
{{ output.data['text/html'] }}
</div>
{% endblock data_html %}
