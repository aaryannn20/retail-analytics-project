# powerbi_data_fixed.py - Corrected version
# Author: Aryan Mishra

import pandas as pd
import numpy as np
import os

def create_powerbi_datasets():
    """Create optimized datasets for Power BI dashboard - FIXED VERSION"""
    
    print("🔄 Preparing data for Power BI...")
    
    # Create PowerBI folder
    powerbi_dir = 'data/powerbi'
    os.makedirs(powerbi_dir, exist_ok=True)
    
    # Load generated data
    try:
        df = pd.read_csv('data/raw/retail_sales_data.csv')
        segments = pd.read_csv('data/processed/customer_segments.csv')
        print("✅ Source data loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run retail_analytics.py first to generate the data")
        return
    
    # Convert transaction_date to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # 1. Sales Fact Table (Main transaction data)
    print("📊 Creating sales fact table...")
    sales_fact = df.copy()
    
    # Add time dimensions
    sales_fact['year'] = sales_fact['transaction_date'].dt.year
    sales_fact['month'] = sales_fact['transaction_date'].dt.month
    sales_fact['quarter'] = sales_fact['transaction_date'].dt.quarter
    sales_fact['week'] = sales_fact['transaction_date'].dt.isocalendar().week
    sales_fact['day_name'] = sales_fact['transaction_date'].dt.day_name()
    sales_fact['month_name'] = sales_fact['transaction_date'].dt.month_name()
    sales_fact['is_weekend'] = sales_fact['transaction_date'].dt.dayofweek >= 5
    
    # Add business metrics
    np.random.seed(42)  # For consistent results
    sales_fact['profit_margin'] = np.random.uniform(0.15, 0.35, len(sales_fact))
    sales_fact['profit'] = sales_fact['total_revenue'] * sales_fact['profit_margin']
    sales_fact['discount_amount'] = sales_fact['unit_price'] * sales_fact['quantity'] * sales_fact['discount']
    sales_fact['gross_revenue'] = sales_fact['total_revenue'] + sales_fact['discount_amount']
    
    # Add customer value tiers
    customer_totals = df.groupby('customer_id')['total_revenue'].sum()
    customer_tiers = pd.cut(customer_totals, 
                           bins=[0, 500, 1500, 5000, float('inf')],
                           labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    tier_mapping = customer_tiers.to_dict()
    sales_fact['customer_tier'] = sales_fact['customer_id'].map(tier_mapping)
    
    sales_fact.to_csv(f'{powerbi_dir}/sales_fact.csv', index=False)
    print(f"✅ Sales fact table created: {len(sales_fact):,} records")
    
    # 2. Customer Dimension Table
    print("👥 Creating customer dimension table...")
    customer_dim = segments.reset_index()
    customer_dim['customer_id'] = customer_dim['customer_id'].astype(str)
    
    # Add readable segment names (handle missing segments gracefully)
    if customer_dim['segment'].max() == 1:
        segment_mapping = {0: 'Discount Driven', 1: 'High Value Loyalists'}
    else:
        # Handle case where there might be more segments
        unique_segments = sorted(customer_dim['segment'].unique())
        segment_names = ['Discount Driven', 'High Value Loyalists', 'Regular Customers', 'Premium Buyers']
        segment_mapping = {seg: segment_names[i] if i < len(segment_names) else f'Segment_{seg}' 
                          for i, seg in enumerate(unique_segments)}
    
    customer_dim['segment_name'] = customer_dim['segment'].map(segment_mapping)
    
    # Add customer performance metrics
    customer_dim['customer_tier'] = customer_dim['customer_id'].astype(int).map(tier_mapping)
    customer_dim['revenue_per_order'] = customer_dim['total_spent'] / customer_dim['order_frequency']
    customer_dim['recency_category'] = pd.cut(customer_dim['recency'], 
                                             bins=[0, 30, 90, 180, float('inf')],
                                             labels=['Very Recent', 'Recent', 'Moderate', 'Inactive'])
    
    # Calculate customer scores (0-100)
    customer_dim['clv_score'] = (customer_dim['total_spent'] / customer_dim['total_spent'].max() * 100).round(0)
    customer_dim['frequency_score'] = (customer_dim['order_frequency'] / customer_dim['order_frequency'].max() * 100).round(0)
    customer_dim['recency_score'] = ((customer_dim['recency'].max() - customer_dim['recency']) / customer_dim['recency'].max() * 100).round(0)
    
    customer_dim.to_csv(f'{powerbi_dir}/customer_dimension.csv', index=False)
    print(f"✅ Customer dimension created: {len(customer_dim):,} customers")
    
    # 3. Product Dimension Table
    print("📦 Creating product dimension table...")
    product_dim = df.groupby('product_category').agg({
        'total_revenue': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'unit_price': ['min', 'max', 'mean'],
        'discount': ['mean', 'max'],
        'customer_id': 'nunique'
    }).round(2)
    
    # Flatten column names
    product_dim.columns = ['total_revenue', 'avg_revenue_per_transaction', 'transaction_count', 
                          'total_quantity_sold', 'min_price', 'max_price', 'avg_price',
                          'avg_discount', 'max_discount', 'unique_customers']
    product_dim = product_dim.reset_index()
    
    # Add performance metrics
    total_rev = product_dim['total_revenue'].sum()
    product_dim['revenue_percentage'] = (product_dim['total_revenue'] / total_rev * 100).round(1)
    product_dim['avg_items_per_transaction'] = product_dim['total_quantity_sold'] / product_dim['transaction_count']
    product_dim['customer_penetration'] = (product_dim['unique_customers'] / df['customer_id'].nunique() * 100).round(1)
    
    # Add category performance ranking
    product_dim['revenue_rank'] = product_dim['total_revenue'].rank(method='dense', ascending=False).astype(int)
    
    product_dim.to_csv(f'{powerbi_dir}/product_dimension.csv', index=False)
    print(f"✅ Product dimension created: {len(product_dim)} categories")
    
    # 4. Time Dimension Table
    print("📅 Creating time dimension table...")
    date_range = pd.date_range(start=df['transaction_date'].min(), 
                              end=df['transaction_date'].max(), freq='D')
    
    time_dim = pd.DataFrame({
        'date': date_range,
        'year': date_range.year,
        'month': date_range.month,
        'quarter': date_range.quarter,
        'week': date_range.isocalendar().week,
        'day_of_month': date_range.day,
        'day_of_week': date_range.dayofweek + 1,
        'day_name': date_range.day_name(),
        'month_name': date_range.month_name(),
        'quarter_name': 'Q' + date_range.quarter.astype(str),
        'is_weekend': date_range.dayofweek >= 5,
        'is_holiday_season': date_range.month.isin([11, 12]),  # Nov, Dec
        'is_summer': date_range.month.isin([6, 7, 8])  # Jun, Jul, Aug
    })
    
    time_dim.to_csv(f'{powerbi_dir}/time_dimension.csv', index=False)
    print(f"✅ Time dimension created: {len(time_dim)} days")
    
    # 5. KPI Summary Table
    print("📈 Creating KPI summary table...")
    # Calculate Pareto analysis
    customer_totals_sorted = customer_totals.sort_values(ascending=False)
    top_20_pct_count = int(len(customer_totals_sorted) * 0.2)
    top_20_revenue = customer_totals_sorted.iloc[:top_20_pct_count].sum()
    pareto_percentage = (top_20_revenue / customer_totals_sorted.sum() * 100)
    
    kpi_summary = pd.DataFrame({
        'metric': [
            'Total Revenue',
            'Total Customers', 
            'Average Order Value',
            'Customer Lifetime Value',
            'Top 20% Revenue Contribution',
            'Average Customer Lifetime (Days)',
            'Total Transactions',
            'High Value Customers',
            'Discount Driven Customers'
        ],
        'value': [
            df['total_revenue'].sum(),
            df['customer_id'].nunique(),
            df['total_revenue'].mean(),
            segments['total_spent'].mean(),
            pareto_percentage,
            segments['customer_lifetime'].mean(),
            len(df),
            len(segments[segments['segment'] == 1]) if 1 in segments['segment'].values else 0,
            len(segments[segments['segment'] == 0]) if 0 in segments['segment'].values else 0
        ],
        'format': [
            'Currency',
            'Number',
            'Currency', 
            'Currency',
            'Percentage',
            'Number',
            'Number',
            'Number',
            'Number'
        ]
    })
    
    kpi_summary.to_csv(f'{powerbi_dir}/kpi_summary.csv', index=False)
    print(f"✅ KPI summary created")
    
    # 6. Monthly Summary for Trends - FIXED VERSION
    print("📊 Creating monthly summary...")
    
    # Create a working copy with required columns
    df_monthly = df.copy()
    df_monthly['year'] = df_monthly['transaction_date'].dt.year
    df_monthly['month'] = df_monthly['transaction_date'].dt.month
    df_monthly['month_name'] = df_monthly['transaction_date'].dt.month_name()
    
    monthly_summary = df_monthly.groupby(['year', 'month', 'month_name']).agg({
        'total_revenue': ['sum', 'count'],
        'customer_id': 'nunique',
        'quantity': 'sum',
        'discount': 'mean'
    }).round(2)
    
    # Flatten column names
    monthly_summary.columns = ['revenue', 'transactions', 'customers', 'units_sold', 'avg_discount']
    monthly_summary = monthly_summary.reset_index()
    monthly_summary['avg_order_value'] = monthly_summary['revenue'] / monthly_summary['transactions']
    monthly_summary['date'] = pd.to_datetime(monthly_summary[['year', 'month']].assign(day=1))
    
    # Add growth rates
    monthly_summary = monthly_summary.sort_values('date')
    monthly_summary['revenue_growth'] = monthly_summary['revenue'].pct_change() * 100
    monthly_summary['customer_growth'] = monthly_summary['customers'].pct_change() * 100
    
    monthly_summary.to_csv(f'{powerbi_dir}/monthly_summary.csv', index=False)
    print(f"✅ Monthly summary created: {len(monthly_summary)} months")
    
    # Generate Power BI connection guide
    create_connection_guide(powerbi_dir)
    
    print(f"\n🎉 Power BI datasets created successfully!")
    print(f"📁 Files created in {powerbi_dir}/ folder:")
    print(f"   • sales_fact.csv - Main transaction data ({len(sales_fact):,} records)")
    print(f"   • customer_dimension.csv - Customer segments and metrics ({len(customer_dim):,} customers)")
    print(f"   • product_dimension.csv - Product category analysis ({len(product_dim)} categories)")
    print(f"   • time_dimension.csv - Date/time hierarchy ({len(time_dim)} days)")
    print(f"   • kpi_summary.csv - Key performance indicators")
    print(f"   • monthly_summary.csv - Monthly trend data ({len(monthly_summary)} months)")
    print(f"   • powerbi_connection_guide.txt - Setup instructions")

def create_connection_guide(powerbi_dir):
    """Create a guide for connecting data to Power BI"""
    guide_content = """
POWER BI CONNECTION GUIDE
========================

FILES TO LOAD:
1. sales_fact.csv (main transaction table)
2. customer_dimension.csv 
3. product_dimension.csv
4. time_dimension.csv
5. kpi_summary.csv
6. monthly_summary.csv

LOADING STEPS:
1. Open Power BI Desktop
2. Click "Get Data" → "Text/CSV" 
3. Navigate to the powerbi folder
4. Load each CSV file above

RELATIONSHIPS TO CREATE:
- sales_fact[customer_id] ↔ customer_dimension[customer_id]
- sales_fact[product_category] ↔ product_dimension[product_category]
- sales_fact[transaction_date] ↔ time_dimension[date]

KEY MEASURES (DAX):
Total Revenue = SUM(sales_fact[total_revenue])
Total Customers = DISTINCTCOUNT(sales_fact[customer_id])
AOV = DIVIDE([Total Revenue], COUNT(sales_fact[customer_id]))

RECOMMENDED VISUALS:
- KPI Cards: Revenue, Customers, AOV
- Line Chart: Monthly trends
- Bar Chart: Category performance  
- Scatter Plot: Customer segmentation
- Donut Chart: Revenue distribution

Author: Aryan Mishra
Project: Retail Analytics & Customer Segmentation
"""
    
    with open(f'{powerbi_dir}/powerbi_connection_guide.txt', 'w', encoding='utf-8') as f:
        f.write(guide_content)

if __name__ == "__main__":
    create_powerbi_datasets()