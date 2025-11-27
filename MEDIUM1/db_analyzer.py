#!/usr/bin/env python3
"""
Database Insights Agent - AI CODEFIX 2025
Analyzes SQLite databases and sends comprehensive reports via email.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
import argparse
import os
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from io import BytesIO

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: reportlab not installed. PDF generation will be skipped.")

# Configuration
TEAM_NAME = "TensorAI Team"
OUTPUT_DIR = "output"

# Email Configuration (to be provided during event)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "your_email@gmail.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "your_app_password")


class DatabaseAnalyzer:
    """Main class for database analysis and reporting."""
    
    def __init__(self, db_path):
        """Initialize the analyzer with database path."""
        self.db_path = db_path
        self.conn = None
        self.tables = []
        self.table_data = {}
        self.insights = []
        self.charts = []
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Set visualization style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 300
        
    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✓ Connected to database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"✗ Error connecting to database: {e}")
            return False
    
    def discover_schema(self):
        """Discover all tables and their schemas."""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.tables = [row[0] for row in cursor.fetchall()]
        print(f"✓ Found {len(self.tables)} tables: {', '.join(self.tables)}")
        
        # Get schema for each table
        self.schema_info = {}
        for table in self.tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            self.schema_info[table] = {
                'columns': [(col[1], col[2]) for col in columns],
                'primary_key': [col[1] for col in columns if col[5] == 1]
            }
            
            # Check for foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            fk_list = cursor.fetchall()
            self.schema_info[table]['foreign_keys'] = fk_list
            
        return self.schema_info
    
    def load_data(self):
        """Load data from all tables into pandas DataFrames."""
        for table in self.tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", self.conn)
                self.table_data[table] = df
                print(f"✓ Loaded {len(df)} rows from '{table}'")
            except Exception as e:
                print(f"✗ Error loading table {table}: {e}")
                self.table_data[table] = pd.DataFrame()
        
        return self.table_data
    
    def analyze_data(self):
        """Perform comprehensive data analysis."""
        self.analysis_results = {}
        
        for table, df in self.table_data.items():
            if df.empty:
                continue
                
            analysis = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            }
            
            # Numeric column statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                analysis['numeric_stats'] = df[numeric_cols].describe().to_dict()
            
            # Categorical column statistics
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                analysis['categorical_stats'] = {}
                for col in categorical_cols:
                    analysis['categorical_stats'][col] = {
                        'unique_values': df[col].nunique(),
                        'top_values': df[col].value_counts().head(5).to_dict()
                    }
            
            # Date column detection and analysis
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        analysis['date_columns'] = analysis.get('date_columns', [])
                        analysis['date_columns'].append(col)
                    except:
                        pass
            
            self.analysis_results[table] = analysis
            
        return self.analysis_results
    
    def generate_insights(self):
        """Generate key insights from the data."""
        self.insights = []
        total_records = sum(len(df) for df in self.table_data.values())
        
        self.insights.append(f"The database contains {len(self.tables)} tables with a total of {total_records:,} records.")
        
        for table, analysis in self.analysis_results.items():
            df = self.table_data[table]
            
            # Data quality insight
            null_cols = [col for col, pct in analysis['null_percentage'].items() if pct > 0]
            if null_cols:
                self.insights.append(f"Table '{table}' has missing values in {len(null_cols)} columns.")
            
            # Numeric insights
            if 'numeric_stats' in analysis:
                for col, stats in analysis['numeric_stats'].items():
                    if 'mean' in stats and 'max' in stats:
                        self.insights.append(
                            f"In '{table}', the '{col}' column has an average of {stats['mean']:.2f} "
                            f"(range: {stats['min']:.2f} to {stats['max']:.2f})."
                        )
                        break  # Just one numeric insight per table
            
            # Categorical insights
            if 'categorical_stats' in analysis:
                for col, stats in analysis['categorical_stats'].items():
                    if stats['unique_values'] > 1:
                        top_value = list(stats['top_values'].keys())[0] if stats['top_values'] else None
                        if top_value:
                            self.insights.append(
                                f"In '{table}', the most common '{col}' is '{top_value}'."
                            )
                            break  # Just one categorical insight per table
        
        # Limit to top 10 insights
        self.insights = self.insights[:10]
        print(f"✓ Generated {len(self.insights)} insights")
        
        return self.insights
    
    def create_visualizations(self):
        """Create at least 3 professional visualizations."""
        self.charts = []
        chart_count = 0
        
        for table, df in self.table_data.items():
            if df.empty or chart_count >= 3:
                continue
            
            # Chart 1: Bar chart for categorical data or row counts
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0 and chart_count < 3:
                col = categorical_cols[0]
                if df[col].nunique() <= 15:  # Limit categories for readability
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts = df[col].value_counts().head(10)
                    sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
                    ax.set_title(f"Distribution of {col} in {table}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Count", fontsize=12)
                    ax.set_ylabel(col, fontsize=12)
                    plt.tight_layout()
                    
                    chart_path = os.path.join(OUTPUT_DIR, f"chart{chart_count + 1}_distribution.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.charts.append(chart_path)
                    chart_count += 1
                    print(f"✓ Created chart: {chart_path}")
            
            # Chart 2: Numeric distribution (histogram)
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0 and chart_count < 3:
                col = numeric_cols[0]
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True, color="steelblue", ax=ax)
                ax.set_title(f"Distribution of {col} in {table}", fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                plt.tight_layout()
                
                chart_path = os.path.join(OUTPUT_DIR, f"chart{chart_count + 1}_histogram.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.charts.append(chart_path)
                chart_count += 1
                print(f"✓ Created chart: {chart_path}")
            
            # Chart 3: Correlation heatmap or pie chart
            if len(numeric_cols) >= 2 and chart_count < 3:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0,
                           square=True, linewidths=0.5, ax=ax)
                ax.set_title(f"Correlation Matrix for {table}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                chart_path = os.path.join(OUTPUT_DIR, f"chart{chart_count + 1}_correlation.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.charts.append(chart_path)
                chart_count += 1
                print(f"✓ Created chart: {chart_path}")
            elif len(categorical_cols) > 0 and chart_count < 3:
                col = categorical_cols[0]
                if df[col].nunique() <= 8:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    value_counts = df[col].value_counts()
                    colors_pie = sns.color_palette("husl", len(value_counts))
                    ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                          colors=colors_pie, startangle=90)
                    ax.set_title(f"Distribution of {col} in {table}", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    chart_path = os.path.join(OUTPUT_DIR, f"chart{chart_count + 1}_pie.png")
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.charts.append(chart_path)
                    chart_count += 1
                    print(f"✓ Created chart: {chart_path}")
        
        # If we still don't have 3 charts, create a summary chart
        if chart_count < 3:
            fig, ax = plt.subplots(figsize=(10, 6))
            table_sizes = {table: len(df) for table, df in self.table_data.items()}
            sns.barplot(x=list(table_sizes.values()), y=list(table_sizes.keys()), 
                       palette="rocket", ax=ax)
            ax.set_title("Records per Table", fontsize=14, fontweight='bold')
            ax.set_xlabel("Number of Records", fontsize=12)
            ax.set_ylabel("Table", fontsize=12)
            plt.tight_layout()
            
            chart_path = os.path.join(OUTPUT_DIR, f"chart{chart_count + 1}_table_summary.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.charts.append(chart_path)
            chart_count += 1
            print(f"✓ Created chart: {chart_path}")
        
        return self.charts
    
    def generate_html_report(self):
        """Generate a professional HTML report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_records = sum(len(df) for df in self.table_data.values())
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Analysis Report - {TEAM_NAME}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0;
            font-size: 2em;
        }}
        .stat-card p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .insight {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Database Analysis Report</h1>
        <p>Generated by {TEAM_NAME} | AI CODEFIX 2025</p>
        <p>Analysis Date: {timestamp}</p>
    </div>

    <div class="section">
        <h2>📈 Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{len(self.tables)}</h3>
                <p>Tables</p>
            </div>
            <div class="stat-card">
                <h3>{total_records:,}</h3>
                <p>Total Records</p>
            </div>
            <div class="stat-card">
                <h3>{len(self.insights)}</h3>
                <p>Key Insights</p>
            </div>
            <div class="stat-card">
                <h3>{len(self.charts)}</h3>
                <p>Visualizations</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>💡 Key Insights</h2>
        {"".join(f'<div class="insight">{insight}</div>' for insight in self.insights)}
    </div>

    <div class="section">
        <h2>📋 Table Overview</h2>
        <table>
            <tr>
                <th>Table Name</th>
                <th>Columns</th>
                <th>Records</th>
                <th>Data Quality</th>
            </tr>
"""
        
        for table, analysis in self.analysis_results.items():
            null_pct = sum(analysis['null_percentage'].values()) / len(analysis['null_percentage']) if analysis['null_percentage'] else 0
            quality = "Excellent" if null_pct < 1 else "Good" if null_pct < 5 else "Needs Review"
            html_content += f"""
            <tr>
                <td><strong>{table}</strong></td>
                <td>{analysis['column_count']}</td>
                <td>{analysis['row_count']:,}</td>
                <td>{quality}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>

    <div class="section">
        <h2>📊 Visualizations</h2>
"""
        
        for i, chart_path in enumerate(self.charts, 1):
            chart_name = os.path.basename(chart_path)
            html_content += f"""
        <div class="chart-container">
            <h3>Chart {i}</h3>
            <img src="{chart_path}" alt="Chart {i}">
        </div>
"""
        
        html_content += f"""
    </div>

    <div class="footer">
        <p>Generated by {TEAM_NAME} | AI CODEFIX 2025</p>
        <p>Database: {self.db_path}</p>
    </div>
</body>
</html>
"""
        
        report_path = os.path.join(OUTPUT_DIR, "report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Generated HTML report: {report_path}")
        return report_path
    
    def generate_pdf_report(self):
        """Generate a PDF report using reportlab."""
        if not PDF_AVAILABLE:
            print("✗ PDF generation skipped (reportlab not installed)")
            return None
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_records = sum(len(df) for df in self.table_data.values())
        
        pdf_path = os.path.join(OUTPUT_DIR, "report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#667eea')
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#667eea')
        )
        
        story = []
        
        # Title
        story.append(Paragraph("📊 Database Analysis Report", title_style))
        story.append(Paragraph(f"Generated by {TEAM_NAME} | AI CODEFIX 2025", styles['Normal']))
        story.append(Paragraph(f"Analysis Date: {timestamp}", styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("📈 Executive Summary", heading_style))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Tables', str(len(self.tables))],
            ['Total Records', f'{total_records:,}'],
            ['Key Insights', str(len(self.insights))],
            ['Visualizations', str(len(self.charts))]
        ]
        summary_table = Table(summary_data, colWidths=[200, 200])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Key Insights
        story.append(Paragraph("💡 Key Insights", heading_style))
        for insight in self.insights:
            story.append(Paragraph(f"• {insight}", styles['Normal']))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 20))
        
        # Charts
        story.append(Paragraph("📊 Visualizations", heading_style))
        for i, chart_path in enumerate(self.charts, 1):
            if os.path.exists(chart_path):
                story.append(Paragraph(f"Chart {i}", styles['Normal']))
                img = Image(chart_path, width=450, height=300)
                story.append(img)
                story.append(Spacer(1, 20))
        
        doc.build(story)
        print(f"✓ Generated PDF report: {pdf_path}")
        return pdf_path
    
    def send_email(self, recipient_email, report_path=None):
        """Send the analysis report via email."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_records = sum(len(df) for df in self.table_data.values())
        
        # Create message
        msg = MIMEMultipart('mixed')
        msg['Subject'] = f"Database Analysis Report - {TEAM_NAME}"
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        
        # Email body
        insights_text = "\n".join(f"  {i+1}. {insight}" for i, insight in enumerate(self.insights[:5]))
        
        body = f"""
Dear Recipient,

Please find the automated database analysis report below.

=== DATABASE SUMMARY ===
- Total Tables: {len(self.tables)}
- Total Records: {total_records:,}
- Analysis Date: {timestamp}

=== KEY INSIGHTS ===
{insights_text}

=== ATTACHMENTS ===
- Full HTML/PDF report attached
- {len(self.charts)} visualization charts attached

Best regards,
{TEAM_NAME}
AI CODEFIX 2025
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach report
        if report_path and os.path.exists(report_path):
            with open(report_path, 'rb') as f:
                attachment = MIMEApplication(f.read())
                attachment.add_header('Content-Disposition', 'attachment', 
                                     filename=os.path.basename(report_path))
                msg.attach(attachment)
        
        # Attach charts
        for chart_path in self.charts:
            if os.path.exists(chart_path):
                with open(chart_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', 
                                  filename=os.path.basename(chart_path))
                    msg.attach(img)
        
        # Send email
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg)
            print(f"✓ Email sent successfully to {recipient_email}")
            return True
        except Exception as e:
            print(f"✗ Failed to send email: {e}")
            print("  Note: Email credentials may need to be configured.")
            return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Database Insights Agent - AI CODEFIX 2025')
    parser.add_argument('--db', required=True, help='Path to SQLite database file')
    parser.add_argument('--email', help='Recipient email address')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  DATABASE INSIGHTS AGENT - AI CODEFIX 2025")
    print(f"  Team: {TEAM_NAME}")
    print("=" * 60)
    print()
    
    # Initialize analyzer
    analyzer = DatabaseAnalyzer(args.db)
    
    try:
        # Step 1: Connect to database
        print("\n[1/7] Connecting to database...")
        if not analyzer.connect():
            sys.exit(1)
        
        # Step 2: Discover schema
        print("\n[2/7] Discovering database schema...")
        analyzer.discover_schema()
        
        # Step 3: Load data
        print("\n[3/7] Loading data from tables...")
        analyzer.load_data()
        
        # Step 4: Analyze data
        print("\n[4/7] Analyzing data...")
        analyzer.analyze_data()
        
        # Step 5: Generate insights
        print("\n[5/7] Generating insights...")
        analyzer.generate_insights()
        
        # Step 6: Create visualizations
        print("\n[6/7] Creating visualizations...")
        analyzer.create_visualizations()
        
        # Step 7: Generate reports
        print("\n[7/7] Generating reports...")
        html_report = analyzer.generate_html_report()
        pdf_report = analyzer.generate_pdf_report()
        
        # Send email if recipient provided
        if args.email:
            print(f"\nSending email to {args.email}...")
            report_to_send = pdf_report if pdf_report else html_report
            analyzer.send_email(args.email, report_to_send)
        
        print("\n" + "=" * 60)
        print("  ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\nOutput files saved to: {os.path.abspath(OUTPUT_DIR)}/")
        print("\nGenerated files:")
        for chart in analyzer.charts:
            print(f"  - {chart}")
        print(f"  - {html_report}")
        if pdf_report:
            print(f"  - {pdf_report}")
        
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
