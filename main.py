from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.secret_key = "ai_project"

@app.route('/', methods = ['GET', 'POST'])
def landing_page():
    print("here")
    return render_template('index.html')