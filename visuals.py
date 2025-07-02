# visuals.py
def render_metric_card(title: str, value: str, style_class: str) -> str:
    return f"""
    <div class="metric-card {style_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """
