#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import nltk


# In[3]:


nltk.download('vader_lexicon')


# In[6]:


apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')


# In[7]:


#Data Cleaning
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]
reviews_df.dropna(subset=['Translated_Review'], inplace=True)


# In[8]:


merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')


# In[9]:


apps_df['Reviews'] = apps_df['Reviews'].astype(int)
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)


# In[10]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return np.nan


# In[11]:


apps_df['Size'] = apps_df['Size'].apply(convert_size)


# In[12]:


apps_df['Log_Installs'] = np.log1p(apps_df['Installs'])
apps_df['Log_Reviews'] = np.log1p(apps_df['Reviews'])


# In[13]:


def rating_group(rating):
    if rating >= 4:
        return 'Top rated'
    elif rating >= 3:
        return 'Above average'
    elif rating >= 2:
        return 'Average'
    else:
        return 'Below average'

apps_df['Rating_Group'] = apps_df['Rating'].apply(rating_group)


# In[14]:


apps_df['Revenue'] = apps_df['Price'] * apps_df['Installs']


# In[16]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[17]:


apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Year'] = apps_df['Last Updated'].dt.year


# In[18]:


import plotly.express as px
import plotly.io as pio
import webbrowser
import os
html_files_path = "./"
# Make sure the directory exists
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)
plot_containers = ""
# Save each Plotly figure to an HTML file
def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    # Append the plot and its insight to plot_containers
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')

# Define your plots
plot_width = 400
plot_height = 300
plot_bg_color = 'black'
text_color = 'white'
title_font = {'size': 16}
axis_font = {'size': 12}


# In[19]:


# Category Analysis Plot
category_counts = apps_df['Category'].value_counts().nlargest(10)
fig1 = px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Category', 'y': 'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=plot_width,
    height=plot_height
)
fig1.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig1.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig1, "category_analysis.html", "The top categories on the Play Store are dominated by tools, entertainment, and productivity apps. This suggests users are looking for apps that either provide utility or offer leisure activities.")


# In[20]:


# Type Analysis Plot
type_counts = apps_df['Type'].value_counts()
fig2 = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=plot_width,
    height=plot_height
)
fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig2, "type_analysis.html", "Most apps on the Play Store are free, indicating a strategy to attract users first and monetize through ads or in-app purchases.")


# In[21]:


# Rating Distribution Plot
fig3 = px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=plot_width,
    height=plot_height
)
fig3.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig3, "rating_distribution.html", "Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users.")


# In[22]:


#sentiment distribution
sentiment_counts = reviews_df['Sentiment_Score'].value_counts()
fig4 = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=plot_width,
    height=plot_height
)
fig4.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig4.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig4, "sentiment_distribution.html", "Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments.")


# In[23]:


# Installs by Category Plot
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5 = px.bar(
    x=installs_by_category.values,
    y=installs_by_category.index,
    orientation='h',
    labels={'x': 'Installs', 'y': 'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=plot_width,
    height=plot_height
)
fig5.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig5.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig5, "installs_by_category.html", "The categories with the most installs are social and communication apps, which reflects their broad appeal and daily usage.")


# In[24]:


# Updates Per Year Plot
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig6, "updates_per_year.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")


# In[25]:


# Revenue by Category Plot
revenue_by_category = apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7 = px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x': 'Category', 'y': 'Revenue'},
    title='Revenue by Category',
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=plot_width,
    height=plot_height
)
fig7.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig7.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig7, "revenue_by_category.html", "Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential.")


# In[26]:


# Genre Count Plot
genre_counts = apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)
fig8 = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Count'},
    title='Top Genres',
    color=genre_counts.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=plot_width,
    height=plot_height
)
fig8.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig8.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig8, "genres_counts.html", "Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games.")


# In[27]:


# Ratings for Paid vs Free Apps
fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Ratings for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=plot_width,
    height=plot_height
)
fig10.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig10, "ratings_paid_free.html", "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for.")


# In[30]:


#Revenue vs installs
paid_apps = apps_df[apps_df['Price'] > 0]
fig11 = px.scatter(
    paid_apps,
    x='Installs',
    y='Revenue',
    color='Category',
    title='Revenue vs Installs for Paid Apps',
    log_x=True,  # Log scale for better visualization
    log_y=True,
    opacity=0.7,
    trendline='ols',  # Add trendline
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig11.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title='Number of Installs', title_font=axis_font),
    yaxis=dict(title='Revenue', title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)

save_plot_as_html(fig11, "revenue_vs_installs.html", "Revenue tends to increase with installs for paid apps, but variability exists across categories.")


# In[31]:


from datetime import datetime
import pytz
# Get current time in IST
ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist).time()
# Check if the current time is within the allowed range (6 PM - 8 PM IST)
if current_time >= datetime.strptime("18:00", "%H:%M").time() and current_time <= datetime.strptime("20:00", "%H:%M").time():
    # Filter categories based on conditions
    filtered_apps = apps_df[~apps_df['Category'].str.startswith(('A', 'C', 'G', 'S'))]
    top_categories = filtered_apps.groupby('Category')['Installs'].sum().nlargest(5).reset_index()
    filtered_apps = filtered_apps[filtered_apps['Category'].isin(top_categories['Category'])]
    filtered_apps['Highlight'] = filtered_apps['Installs'] > 1_000_000
    # Create Choropleth map
    fig12 = px.choropleth(
        filtered_apps,
        locations='Country',
        locationmode='country names',
        color='Category',
        hover_name='Category',
        hover_data=['Installs'],
        title='Global Installs by Category',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig12.update_layout(
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=plot_bg_color,
        font_color=text_color,
        title_font=title_font,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    save_plot_as_html(fig12, "global_installs_map.html", "Visualizing the global distribution of installs across the top app categories, excluding those starting with A, C, G, or S.")
else:
    print("Choropleth map not displayed outside 6 PM - 8 PM IST.")


# In[33]:


# Get current time in IST
ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist).time()
if current_time >= datetime.strptime("18:00", "%H:%M").time() and current_time <= datetime.strptime("21:00", "%H:%M").time():
    # Filter data based on conditions
    filtered_apps = apps_df[
        (apps_df['Category'].str.startswith(('E', 'C', 'B')))
        & (~apps_df['App'].str.startswith(('X', 'Y', 'Z')))
        & (apps_df['Reviews'] > 500)
    ]
    # Aggregate installs by month and category
    filtered_apps['Month'] = filtered_apps['Date'].dt.to_period('M')
    installs_trend = filtered_apps.groupby(['Month', 'Category'])['Installs'].sum().reset_index()
    # Calculate month-over-month percentage change
    installs_trend['Growth'] = installs_trend.groupby('Category')['Installs'].pct_change()
    installs_trend['Highlight'] = installs_trend['Growth'] > 0.2  # Highlight if growth > 20%
    # Create line chart
    fig13 = px.line(
        installs_trend,
        x='Month',
        y='Installs',
        color='Category',
        title='Trend of Total Installs Over Time by Category',
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Highlight growth periods
    for category in installs_trend['Category'].unique():
        category_data = installs_trend[installs_trend['Category'] == category]
        highlight_data = category_data[category_data['Highlight']]
        fig_installs_trend.add_traces(px.area(highlight_data, x='Month', y='Installs', opacity=0.3).data)
    
    fig13.update_layout(
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=plot_bg_color,
        font_color=text_color,
        title_font=title_font,
        xaxis=dict(title='Month', title_font=axis_font),
        yaxis=dict(title='Total Installs', title_font=axis_font),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    save_plot_as_html(fig13, "installs_trend_chart.html", "This chart visualizes the monthly trend of total installs across selected app categories, highlighting significant growth periods.")
else:
    print("Time series chart not displayed outside 6 PM - 9 PM IST.")


# In[34]:


# Split plot_containers to handle the last plot properly
plot_containers_split = plot_containers.split('</div>')
if len(plot_containers_split) > 1:
    final_plot = plot_containers_split[-2] + '</div>'
else:
    final_plot = plot_containers  # Use plot_containers as default if splitting isn't sufficient

# HTML template for the dashboard
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Reviews Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container:hover .insights {{
            display: block;
        }}
    </style>
    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
    </div>
    <div class="container">
        {plots}
    </div>
</body>
</html>
"""

# Use these containers to fill in your dashboard HTML
final_html = dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height)

# Save the final dashboard to an HTML file
dashboard_path = os.path.join(html_files_path, "dashboard.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

# Automatically open the generated HTML file in a web browser
webbrowser.open('file://' + os.path.realpath(dashboard_path))


# In[ ]:




