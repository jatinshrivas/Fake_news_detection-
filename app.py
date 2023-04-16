from flask import Flask, escape, request, render_template,url_for
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction
app=Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict')
def predict():
    return render_template("prediction.html")
@app.route('/contact')
def contact():
    return render_template("Contact us.html")
@app.route('/index')
def news():
    return render_template("index.html")
@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        ns=request.form["news"]
        xy=fake_news(ns)
        print(xy[0])
        if xy[0]=="fake":
            return render_template("prediction.html" ,prediction_text="The Given News Headline is not Reliable")
        else:
            return render_template("prediction.html", prediction_text="The Given News Headline is Reliable")
    else:
        return render_template("prediction.html")
app.run()