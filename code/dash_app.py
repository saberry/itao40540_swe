# ##################
# Dash Application #
####################

# The goal is to take the lyrics data and produce the following:
# Sentiment analysis displayed by genre
# Sentiment analysis displayed by week and genre
# Visualizations for sentiment

from dash import Dash, dcc, html, callback
# The dcc function is dash core components
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')

# Load the data 

lyrics = pd.read_feather('~/Documents/teaching/itao40540_swe/all_lyrics_23_24.feather')

lyrics['week'] = pd.to_datetime(lyrics['week']).dt.date

lyrics['score'] = None

# Now we need to get a sentiment score for each song:

for row in range(0, len(lyrics)):
    song = lyrics['lyrics'][row]
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(song)
    compound_score = ss['compound']
    lyrics.loc[row, 'score'] = compound_score

# fig = px.histogram(lyrics['score'])
# fig.show()
# 
# agg_data = lyrics.groupby(['genre', 'week']).agg({'score': 'mean'}).reset_index()
# 
# fig = px.line(agg_data, x='week', y='score', color='genre')
# fig.show()

# Now with our sentiment scores in, 
# we can start to create a dashboard to display the data

# Create a list of genres to use in the dropdown
genres = lyrics['genre'].unique()

# Create a Dash app
app = Dash(__name__)

# Create the layout

app.layout = html.Div([
    # This is the title of the dashboard
    html.H1('Lyrics Sentiment Analysis'),
    html.Div([
        # This is the dropdown label
        html.Label('Choose a genre:'),
        # This is the dropdown with genre options
        dcc.Dropdown(
            id='genre-dropdown',
            options=[{'label': i, 'value': i} for i in genres],
            value='Rock'
        )
    ]),
    html.Div(children = [
        # This is the location for a graph
        # that we will call sentiment-graph-week
        dcc.Graph(
            id='sentiment-graph-week',
            figure={},
            style={'display': 'inline-block'}
        ), 
        dcc.Graph(
            id='sentiment-graph',
            figure={},
            style={'display': 'inline-block'}
        )
    ])
])

# Create the callback
# In other words, the thing that 
# will update the graph

# Outputs always go first!
@callback(
    Output('sentiment-graph', 'figure'),
    Output('sentiment-graph-week', 'figure'),
    Input('genre-dropdown', 'value')
)
def update_graph(selected_genre):
    filtered_lyrics = lyrics[lyrics['genre'] == selected_genre]
    x = filtered_lyrics['score']
    agg_data = filtered_lyrics.groupby(['genre', 'week']).agg({'score': 'mean'}).reset_index()
    agg_data['week'] = pd.to_datetime(agg_data['week'])
    # Now we want to return a 
    # histogram of the sentiment scores
    # for the selected genre
    fig1 = px.histogram(x=x)
    fig1.layout.title = f'Sentiment Analysis for {selected_genre}'
    # And the line graph of sentiment scores
    
    fig2 = px.line(agg_data, x='week', y='score', color='genre')
    fig2.layout.title = f'Average Sentiment for {selected_genre} over Week'
    return [fig1, fig2]
  
# Run the app

app.run_server(debug=False)
