import os
import pandas as pd
import numpy as np
import neattext.functions as nfx
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dashboard  # Import full module to avoid circular imports

app = Flask(__name__)

def getcosinemat(df):
    countvect = CountVectorizer()
    cvmat=countvect.fit_transform(df['Clean_title'])
    return cvmat

def getcleantitle(df):
    df['Clean_title'] = df['course_title'].apply(nfx.remove_stopwords)
    df['Clean_title'] = df['Clean_title'].apply(nfx.remove_special_characters)
    return df

def readdata():
    file_path = 'UdemyCleanedTitle.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
    return pd.read_csv(file_path)

def recommend_course(df, title, cosine_mat, numrec=6):
    if title not in df['course_title'].values:
        return pd.DataFrame()  # Empty result

    course_index = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    index = course_index.get(title)

    if index is None:
        return pd.DataFrame()

    scores = list(enumerate(cosine_mat[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:numrec+1]

    selected_course_index = [i[0] for i in sorted_scores]
    selected_course_score = [i[1] for i in sorted_scores]

    rec_df = df.iloc[selected_course_index].copy()
    rec_df.loc[:, 'Similarity_Score'] = selected_course_score

    return rec_df[['course_title', 'Similarity_Score', 'url', 'price', 'num_subscribers']]

def searchterm(term, df):
    result_df = df[df['course_title'].str.contains(term, case=False, na=False)]
    return result_df.sort_values(by='num_subscribers', ascending=False).head(6)

def extractfeatures(recdf):
    return list(recdf['url']), list(recdf['course_title']), list(recdf['price'])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        titlename = request.form['course']
        print(titlename)
        try:
            df = readdata()
            df = getcleantitle(df)
            cvmat = getcosinemat(df)
            num_rev=6
            cosine_mat = cosine_similarity(cvmat)
            recdf = recommend_course(df, titlename, cosine_mat, num_rev)
            
            if not recdf.empty:
                coursemap = dict(zip(*extractfeatures(recdf)))
                return render_template('index.html', coursemap=coursemap, coursename=titlename, showtitle=True)
            else:
                return render_template('index.html', showerror=True, coursename=titlename)
        except:
            resultdf = searchterm(titlename, readdata())
            coursemap = dict(zip(*extractfeatures(resultdf))) if not resultdf.empty else {}
            return render_template('index.html', coursemap=coursemap, coursename=titlename, showtitle=True if coursemap else False, showerror=not bool(coursemap))

    return render_template('index.html')

@app.route('/dashboard')
def dashboard_page():
    df = readdata()
    valuecounts = dashboard.getvaluecounts(df)
    levelcounts = dashboard.getlevelcount(df)
    subjectsperlevel = dashboard.getsubjectsperlevel(df)
    yearwiseprofitmap, subscriberscountmap, profitmonthwise, monthwisesub = dashboard.yearwiseprofit(df)

    return render_template('dashboard.html', valuecounts=valuecounts, levelcounts=levelcounts,
                           subjectsperlevel=subjectsperlevel, yearwiseprofitmap=yearwiseprofitmap,
                           subscriberscountmap=subscriberscountmap, profitmonthwise=profitmonthwise, monthwisesub=monthwisesub)

if __name__ == '_main_':
    app.run(debug=True)