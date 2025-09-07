import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration (inline since config.py might not exist)
PROJECT_NAME = "Retail Sales Analysis & Customer Segmentation"
VERSION = "1.0.0"
AUTHOR = "Aryan Mishra"
N_RECORDS = 50000
RANDOM_SEED = 42

# Set up directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'outputs'), exist_ok=True)

# Set style for better visualizations
plt.style.use('default')  # Use default style to avoid seaborn version issues
sns.set_palette("husl")

class RetailAnalytics:
    def __init__(self):
        self.df = None
        self.customer_segments = None
        self.customer_stats = None
        self.segment_names = {}
        self.scaler = StandardScaler()
        
    def generate_sample_data(self, n_records=50000):
        """Generate realistic retail sales data"""
        print(f"🔄 Generating {n_records:,} sample records...")
        
        np.random.seed(RANDOM_SEED)
        
        # Generate base data
        customer_ids = np.random.randint(1000, 9999, n_records)
        
        # Product categories and their price ranges
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty']
        category_prices = {
            'Electronics': (50, 2000),
            'Clothing': (15, 300),
            'Home & Garden': (10, 500),
            'Sports': (20, 800),
            'Books': (5, 50),
            'Beauty': (8, 150)
        }
        
        data = []
        for i in range(n_records):
            # Customer behavior patterns
            customer_type = np.random.choice(['high_spender', 'discount_driven', 'regular'], 
                                           p=[0.2, 0.3, 0.5])
            
            category = np.random.choice(categories)
            min_price, max_price = category_prices[category]
            
            # Adjust pricing based on customer type
            if customer_type == 'high_spender':
                base_price = np.random.uniform(min_price * 0.8, max_price)
                discount = np.random.uniform(0, 0.1)  # Low discount preference
                quantity = np.random.randint(1, 5)
            elif customer_type == 'discount_driven':
                base_price = np.random.uniform(min_price, max_price * 0.7)
                discount = np.random.uniform(0.1, 0.4)  # High discount preference
                quantity = np.random.randint(1, 3)
            else:  # regular
                base_price = np.random.uniform(min_price, max_price * 0.8)
                discount = np.random.uniform(0, 0.2)
                quantity = np.random.randint(1, 3)
            
            # Generate seasonal effects
            month = np.random.randint(1, 13)
            seasonal_multiplier = 1.0
            if month in [11, 12]:  # Holiday season
                seasonal_multiplier = 1.3
            elif month in [6, 7]:  # Summer sales
                seasonal_multiplier = 1.1
            
            final_price = base_price * seasonal_multiplier * (1 - discount)
            revenue = final_price * quantity
            
            # Create random dates within the last 2 years
            start_date = pd.Timestamp('2023-01-01')
            end_date = pd.Timestamp('2024-12-31')
            random_date = start_date + pd.Timedelta(days=np.random.randint(0, (end_date - start_date).days))
            
            data.append({
                'customer_id': customer_ids[i],
                'transaction_date': random_date,
                'product_category': category,
                'quantity': quantity,
                'unit_price': round(base_price, 2),
                'discount': round(discount, 3),
                'total_revenue': round(revenue, 2),
                'customer_type': customer_type
            })
        
        self.df = pd.DataFrame(data)
        self.df['month'] = self.df['transaction_date'].dt.month
        self.df['year'] = self.df['transaction_date'].dt.year
        self.df['day_of_week'] = self.df['transaction_date'].dt.day_name()
        
        print(f"✅ Generated {len(self.df):,} sales records")
        print(f"📅 Date range: {self.df['transaction_date'].min().date()} to {self.df['transaction_date'].max().date()}")
        return self.df
    
    def perform_eda(self):
        """Comprehensive Exploratory Data Analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print(f"\n📊 Dataset Overview:")
        print(f"Total Records: {len(self.df):,}")
        print(f"Unique Customers: {self.df['customer_id'].nunique():,}")
        print(f"Date Range: {self.df['transaction_date'].min().date()} to {self.df['transaction_date'].max().date()}")
        print(f"Total Revenue: ${self.df['total_revenue'].sum():,.2f}")
        print(f"Average Order Value: ${self.df['total_revenue'].mean():.2f}")
        print(f"Product Categories: {', '.join(self.df['product_category'].unique())}")
        
        # Revenue trends by month
        monthly_revenue = self.df.groupby(['year', 'month'])['total_revenue'].sum().reset_index()
        monthly_revenue['date'] = pd.to_datetime(monthly_revenue[['year', 'month']].assign(day=1))
        
        print(f"\n🔝 Top 3 Revenue Months:")
        top_months = monthly_revenue.nlargest(3, 'total_revenue')
        for _, row in top_months.iterrows():
            month_name = pd.to_datetime(f"{row['year']}-{row['month']}-01").strftime('%B %Y')
            print(f"  {month_name}: ${row['total_revenue']:,.2f}")
        
        # Category performance
        print(f"\n📦 Category Performance:")
        category_revenue = self.df.groupby('product_category')['total_revenue'].sum().sort_values(ascending=False)
        for category, revenue in category_revenue.items():
            pct = (revenue / self.df['total_revenue'].sum()) * 100
            print(f"  {category}: ${revenue:,.2f} ({pct:.1f}%)")
        
        return self.df.describe()
    
    def analyze_customer_behavior(self):
        """Analyze customer purchasing patterns"""
        print("\n=== CUSTOMER BEHAVIOR ANALYSIS ===")
        
        # Customer lifetime value and segmentation data
        customer_stats = self.df.groupby('customer_id').agg({
            'total_revenue': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'discount': 'mean',
            'transaction_date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        customer_stats.columns = ['total_spent', 'avg_order_value', 'order_frequency', 
                                'total_items', 'avg_discount', 'first_purchase', 'last_purchase']
        
        # Calculate customer lifetime (days)
        customer_stats['customer_lifetime'] = (
            customer_stats['last_purchase'] - customer_stats['first_purchase']
        ).dt.days
        
        # Calculate recency (days since last purchase)
        customer_stats['recency'] = (
            self.df['transaction_date'].max() - customer_stats['last_purchase']
        ).dt.days
        
        print(f"\n👥 Customer Insights:")
        print(f"Average Customer Lifetime Value: ${customer_stats['total_spent'].mean():.2f}")
        print(f"Average Order Frequency: {customer_stats['order_frequency'].mean():.1f} orders")
        print(f"Average Customer Lifetime: {customer_stats['customer_lifetime'].mean():.0f} days")
        
        # Pareto analysis (80-20 rule)
        customer_stats_sorted = customer_stats.sort_values('total_spent', ascending=False)
        total_revenue = customer_stats_sorted['total_spent'].sum()
        
        # Find top 20% of customers
        top_20_pct_customers = int(len(customer_stats_sorted) * 0.2)
        top_20_revenue_contribution = customer_stats_sorted.iloc[:top_20_pct_customers]['total_spent'].sum()
        top_20_percentage = (top_20_revenue_contribution / total_revenue) * 100
        
        print(f"\n📈 Pareto Analysis:")
        print(f"Top 20% of customers contribute {top_20_percentage:.1f}% of total revenue")
        print(f"Revenue from top 20%: ${top_20_revenue_contribution:,.2f}")
        
        self.customer_stats = customer_stats
        return customer_stats
    
    def perform_customer_segmentation(self):
        """K-Means clustering for customer segmentation"""
        print("\n=== CUSTOMER SEGMENTATION (K-MEANS) ===")
        
        # Prepare features for clustering
        features = self.customer_stats[['total_spent', 'order_frequency', 'avg_order_value', 
                                      'recency', 'avg_discount']].fillna(0)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Find optimal number of clusters using elbow method
        silhouette_scores = []
        k_range = range(2, 8)
        
        print("🔍 Finding optimal number of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            kmeans.fit(features_scaled)
            silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
        
        # Choose optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"✅ Optimal number of clusters: {optimal_k}")
        print(f"📊 Silhouette score: {max(silhouette_scores):.3f}")
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to customer stats
        self.customer_stats['segment'] = cluster_labels
        
        # Define segment names based on characteristics
        segment_names = {}
        for segment in range(optimal_k):
            seg_data = self.customer_stats[self.customer_stats['segment'] == segment]
            avg_spent = seg_data['total_spent'].mean()
            avg_frequency = seg_data['order_frequency'].mean()
            avg_discount = seg_data['avg_discount'].mean()
            
            if avg_spent > self.customer_stats['total_spent'].quantile(0.7):
                if avg_frequency > self.customer_stats['order_frequency'].quantile(0.7):
                    segment_names[segment] = "High Value Loyalists"
                else:
                    segment_names[segment] = "High Spenders"
            elif avg_discount > self.customer_stats['avg_discount'].quantile(0.6):
                segment_names[segment] = "Discount Driven"
            elif avg_frequency == 1 or avg_frequency < self.customer_stats['order_frequency'].quantile(0.3):
                segment_names[segment] = "One-time Buyers"
            else:
                segment_names[segment] = "Regular Customers"
        
        print(f"\n🎯 Customer Segments:")
        for segment in range(optimal_k):
            seg_data = self.customer_stats[self.customer_stats['segment'] == segment]
            seg_name = segment_names[segment]
            print(f"\n{seg_name} (Segment {segment}):")
            print(f"  Size: {len(seg_data):,} customers ({len(seg_data)/len(self.customer_stats)*100:.1f}%)")
            print(f"  Avg Total Spent: ${seg_data['total_spent'].mean():.2f}")
            print(f"  Avg Order Frequency: {seg_data['order_frequency'].mean():.1f}")
            print(f"  Avg Order Value: ${seg_data['avg_order_value'].mean():.2f}")
            print(f"  Avg Recency: {seg_data['recency'].mean():.0f} days")
            print(f"  Avg Discount: {seg_data['avg_discount'].mean():.1%}")
        
        self.segment_names = segment_names
        self.customer_segments = self.customer_stats
        return self.customer_stats
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n=== BUSINESS INSIGHTS & RECOMMENDATIONS ===")
        
        insights = []
        
        # Revenue concentration
        top_20_customers = int(len(self.customer_stats) * 0.2)
        top_20_revenue = self.customer_stats.nlargest(top_20_customers, 'total_spent')['total_spent'].sum()
        total_revenue = self.customer_stats['total_spent'].sum()
        concentration = (top_20_revenue / total_revenue) * 100
        
        insights.append(f"🎯 Revenue Concentration: Top 20% of customers contribute {concentration:.1f}% of revenue")
        
        # Segment insights
        for segment in self.customer_stats['segment'].unique():
            seg_data = self.customer_stats[self.customer_stats['segment'] == segment]
            seg_name = self.segment_names[segment]
            
            if "High Value" in seg_name:
                insights.append(f"💎 {seg_name}: Focus on retention programs and premium services")
            elif "Discount Driven" in seg_name:
                insights.append(f"💰 {seg_name}: Implement targeted discount campaigns and flash sales")
            elif "One-time" in seg_name:
                insights.append(f"🔄 {seg_name}: Develop re-engagement campaigns with special offers")
        
        # Seasonal insights
        monthly_revenue = self.df.groupby('month')['total_revenue'].mean()
        peak_month = monthly_revenue.idxmax()
        
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        insights.append(f"📅 Peak Season: {month_names[peak_month]} shows highest average revenue")
        
        # Category insights
        category_revenue = self.df.groupby('product_category')['total_revenue'].sum().sort_values(ascending=False)
        top_category = category_revenue.index[0]
        category_percentage = (category_revenue.iloc[0] / self.df['total_revenue'].sum()) * 100
        
        insights.append(f"📦 Top Category: {top_category} generates {category_percentage:.1f}% of total revenue")
        
        print(f"\n💡 Key Business Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print(f"\n🚀 Recommended Actions:")
        print("1. 🎯 Implement VIP program for top 20% high-value customers")
        print("2. 💌 Create personalized email campaigns for each customer segment")
        print("3. 📊 Focus inventory and marketing spend on peak months and top categories")
        print("4. 🔄 Launch win-back campaigns for one-time buyers")
        print("5. 💎 Develop premium product lines for high-value segments")
        
        return insights
    
    def create_visualizations(self):
        """Create basic visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        try:
            # 1. Monthly Revenue Trend
            monthly_data = self.df.groupby(['year', 'month'])['total_revenue'].sum().reset_index()
            monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
            
            fig1 = px.line(monthly_data, x='date', y='total_revenue', 
                          title='Monthly Revenue Trend',
                          labels={'total_revenue': 'Revenue ($)', 'date': 'Date'})
            
            # Save as HTML if possible
            try:
                fig1.write_html(os.path.join(DATA_DIR, 'monthly_revenue_trend.html'))
                print("✅ Monthly revenue trend saved")
            except:
                print("⚠️  Could not save visualization file")
            
            # 2. Category Performance
            category_data = self.df.groupby('product_category')['total_revenue'].sum().reset_index()
            
            fig2 = px.bar(category_data, x='product_category', y='total_revenue',
                         title='Revenue by Product Category',
                         labels={'total_revenue': 'Revenue ($)', 'product_category': 'Category'})
            
            try:
                fig2.write_html(os.path.join(DATA_DIR, 'category_performance.html'))
                print("✅ Category performance chart saved")
            except:
                print("⚠️  Could not save visualization file")
            
            # 3. Customer Segments Scatter Plot
            if self.customer_segments is not None:
                fig3 = px.scatter(self.customer_segments, x='total_spent', y='order_frequency', 
                                 color='segment', size='avg_order_value',
                                 title='Customer Segmentation Analysis',
                                 labels={'total_spent': 'Total Spent ($)', 'order_frequency': 'Order Frequency'})
                
                try:
                    fig3.write_html(os.path.join(DATA_DIR, 'customer_segments.html'))
                    print("✅ Customer segmentation chart saved")
                except:
                    print("⚠️  Could not save visualization file")
            
            print("📊 Visualizations created successfully!")
            
        except Exception as e:
            print(f"⚠️  Error creating visualizations: {str(e)}")
    
    def save_results(self):
        """Save analysis results to files"""
        print("\n=== SAVING RESULTS ===")
        
        try:
            # Save raw data
            if self.df is not None:
                self.df.to_csv(os.path.join(DATA_DIR, 'raw', 'retail_sales_data.csv'), index=False)
                print("✅ Raw data saved to CSV")
            
            # Save customer segments
            if self.customer_segments is not None:
                self.customer_segments.to_csv(os.path.join(DATA_DIR, 'processed', 'customer_segments.csv'))
                print("✅ Customer segments saved to CSV")
            
            # Save analysis summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(DATA_DIR, 'outputs', f'analysis_summary_{timestamp}.txt')
            
            with open(summary_file, 'w') as f:
                f.write(f"Retail Analytics Analysis Summary\n")
                f.write(f"Generated on: {datetime.now()}\n")
                f.write(f"Author: {AUTHOR}\n")
                f.write(f"Total records: {len(self.df):,}\n")
                f.write(f"Unique customers: {self.df['customer_id'].nunique():,}\n")
                f.write(f"Total revenue: ${self.df['total_revenue'].sum():,.2f}\n")
                f.write(f"Average order value: ${self.df['total_revenue'].mean():.2f}\n")
                f.write(f"Date range: {self.df['transaction_date'].min().date()} to {self.df['transaction_date'].max().date()}\n")
            
            print(f"✅ Analysis summary saved")
            
        except Exception as e:
            print(f"⚠️  Error saving results: {str(e)}")

def main():
    """Main execution function"""
    print(f"🚀 {PROJECT_NAME}")
    print(f"Version: {VERSION}")
    print(f"Author: {AUTHOR}")
    print("=" * 70)
    
    # Initialize analytics
    analytics = RetailAnalytics()
    
    try:
        # Run complete analysis pipeline
        print("Starting analysis pipeline...")
        
        # Step 1: Generate data
        analytics.generate_sample_data(N_RECORDS)
        
        # Step 2: Exploratory Data Analysis
        analytics.perform_eda()
        
        # Step 3: Customer Behavior Analysis
        analytics.analyze_customer_behavior()
        
        # Step 4: Customer Segmentation
        analytics.perform_customer_segmentation()
        
        # Step 5: Generate Business Insights
        analytics.generate_business_insights()
        
        # Step 6: Create Visualizations
        analytics.create_visualizations()
        
        # Step 7: Save Results
        analytics.save_results()
        
        print("\n" + "=" * 70)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📁 Check the 'data' folder for all output files:")
        print(f"   • Raw data: data/raw/retail_sales_data.csv")
        print(f"   • Customer segments: data/processed/customer_segments.csv")
        print(f"   • Visualizations: data/*.html files")
        print(f"   • Analysis summary: data/outputs/analysis_summary_*.txt")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("🔧 Please check your Python environment and try again.")
        raise

if __name__ == "__main__":
    main()