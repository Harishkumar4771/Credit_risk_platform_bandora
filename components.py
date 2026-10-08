import streamlit as st


def section_title(title, copy=None):
    st.subheader(title)
    if copy:
        st.caption(copy)


def surface(content, class_name="surface-card"):
    st.markdown(f'<div class="{class_name}">{content}</div>', unsafe_allow_html=True)


def badge(text, applies=False):
    css = "badge badge-applies" if applies else "badge badge-no"
    return f'<span class="{css}">{text}</span>'


def step_flow(steps):
    for number, title, copy, details in steps:
        st.markdown(f'<div class="flow-line"><span class="flow-dot">{number}</span><span class="flow-title">{title}</span><p class="flow-copy">{copy}</p></div>', unsafe_allow_html=True)
        with st.expander("Technical details", expanded=False):
            st.write(details)


def risk_result(probability, threshold):
    high = probability >= threshold
    color = "risk-high" if high else "risk-low"
    verdict = "Flagged as higher risk" if high else "Lower risk"
    st.markdown(f'<div class="risk-number {color}">{probability:.1%}</div><div class="risk-verdict {color}">{verdict}</div>', unsafe_allow_html=True)


def outcome_card(row):
    fields = [
        ("Age", f"{int(row.get('Age', 0))}"),
        ("Country", str(row.get("Country", "Unknown"))),
        ("Loan amount", f"€{float(row.get('Amount', 0)):,.0f}"),
        ("Interest", f"{float(row.get('Interest', 0)):.1f}%"),
        ("Duration", f"{int(row.get('LoanDuration', 0))} months"),
        ("Monthly income", f"€{float(row.get('IncomeTotal', 0)):,.0f}"),
        ("Customer", "New" if row.get("NewCreditCustomer") == 1 else "Returning"),
    ]
    columns = st.columns(len(fields))
    for column, (label, value) in zip(columns, fields):
        with column:
            st.markdown(f'<div class="summary-label">{label}</div><div class="summary-value">{value}</div>', unsafe_allow_html=True)
