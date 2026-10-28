import os
import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size:2.5rem;
        font-weight:700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>💰 AI Expense Analyzer</div>", unsafe_allow_html=True)

# Securely retrieve Groq API Key (Checks Streamlit Secrets first, then sidebar/environment)
groq_api_key = ""
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

# Sidebar - API Key and Settings
st.sidebar.header("⚙️ Configuration")
user_api_key = st.sidebar.text_input("Enter Groq API Key (Optional if set in Secrets):", type="password", value=groq_api_key)

if user_api_key:
    groq_api_key = user_api_key

# Initialize Session State for Expenses Data
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Category", "Amount", "Description"])

# Main Form to Add Expenses
st.subheader("➕ Add New Expense")
col1, col2, col3, col4 = st.columns([2, 2, 2, 4])

with col1:
    exp_date = st.date_input("Date", value=datetime.today())
with col2:
    category = st.selectbox("Category", [
        "Food & Dining", "Transportation", "Shopping", 
        "Bills & Utilities", "Entertainment", "Health & Fitness", 
        "Subscriptions", "Others"
    ])
with col3:
    amount = st.number_input("Amount ($)", min_value=0.01, step=1.0)
with col4:
    description = st.text_input("Description", placeholder="e.g., Dinner with friends at Italian restaurant")

if st.button("Add Expense", use_container_width=True):
    if not description.strip():
        st.warning("Please provide a description.")
    else:
        new_entry = pd.DataFrame([{
            "Date": pd.to_datetime(exp_date),
            "Category": category,
            "Amount": amount,
            "Description": description
        }])
        st.session_state.expenses = pd.concat([st.session_state.expenses, new_entry], ignore_index=True)
        st.success("Expense added successfully!")

# Display Data & Analytics if expenses exist
if not st.session_state.expenses.empty:
    df = st.session_state.expenses.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    
    st.markdown("---")
    
    # Overview Metrics
    m1, m2, m3 = st.columns(3)
    total_spent = df["Amount"].sum()
    avg_spent = df["Amount"].mean()
    highest_expense = df.loc[df["Amount"].idxmax()] if len(df) > 0 else None
    
    m1.metric("Total Spending", f"${total_spent:,.2f}")
    m2.metric("Average per Transaction", f"${avg_spent:,.2f}")
    if highest_expense is not None:
        m3.metric("Highest Single Expense", f"${highest_expense['Amount']:,.2f}", highest_expense['Category'])

    # Data Table
    with st.expander("📄 View Recorded Expenses Table", expanded=False):
        st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Visualizations & Spending Trends")
    
    col_chart1, col_chart2 = st.columns(2)
    
    category_df = df.groupby("Category")["Amount"].sum().reset_index()

    with col_chart1:
        # Category breakdown (Pie Chart)
        fig_pie = px.pie(category_df, values='Amount', names='Category', title="Expense Distribution by Category", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        # Spending by Category (Bar Chart)
        fig_bar = px.bar(category_df, x='Category', y='Amount', color='Category', title="Total Spent per Category", text_auto='.2f')
        st.plotly_chart(fig_bar, use_container_width=True)

    # Spending Trends over time
    st.subheader("📈 Spending Trends")
    daily_df = df.groupby("Date")["Amount"].sum().reset_index()
    fig_line = px.line(daily_df, x='Date', y='Amount', markers=True, title="Daily Spending Trend (Highs vs Lows)")
    fig_line.add_hline(y=daily_df["Amount"].mean(), line_dash="dash", line_color="orange", annotation_text="Average Daily Spend")
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 AI Insights & Recommendations (Groq LLM)")

    if st.button("🧠 Analyze Expenses with AI", type="primary"):
        if not groq_api_key:
            st.error("Please enter your Groq API Key in the sidebar or configure it in Streamlit Cloud Secrets.")
        else:
            try:
                client = Groq(api_key=groq_api_key)
                
                # Format expenses data into CSV string prompt
                expense_summary = df.to_csv(index=False)
                
                prompt = f"""
You are an expert financial advisor and AI expense analyzer.
Below is the list of user expenses in CSV format:

{expense_summary}

Please analyze this data and provide a detailed report structured as follows:
1. **Spending Patterns**: Summarize habits, key spending categories, and frequency.
2. **Trend Analysis**: Identify high spending spikes/days and normal/low spending periods.
3. **Optimizations & Strategies**: Give 3-5 actionable and personalized financial recommendations to optimize expenses and save money based on the descriptions provided.
4. **Budgeting Advice**: Suggest a realistic spending threshold or 50/30/20 budget breakdown.

Keep the advice practical, clear, encouraging, and formatted in clean Markdown.
"""

                with st.spinner("Analyzing your spending patterns using Groq AI..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a professional financial planner."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.7,
                    )
                    
                    response = chat_completion.choices[0].message.content
                    st.markdown(response)

            except Exception as e:
                st.error(f"Error communicating with Groq API: {str(e)}")

else:
    st.info("No expenses recorded yet. Add some expenses using the form above to see charts and AI analysis!")
