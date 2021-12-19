import flask
from flask import render_template, url_for, request, Response
from recommender import app
import recommender.emotion
from recommender.movie import m_rec
from recommender.songs import s_rec

@app.route("/")
def home():
    return render_template("home.html", title="Movies And Songs Recommender", home="active")

@app.route("/movies", methods=['POST', 'GET'])
def movies():
    if flask.request.method == 'POST':
        userinput = request.form['movie_name']
        return render_template("movies1.html", title="Movies", movies="active", recmovies=m_rec.get_recommendations(userinput), topmovies=m_rec.topmoviesfn())
    return render_template("movies.html", title="Movies", movies="active", topmovies=m_rec.topmoviesfn())

@app.route("/songs")
def songs():
    #if flask.request.method == 'POST':
        #userinput = request.form['movie_name']
        #return render_template("movies1.html", title="Movies", movies="active", recmovies=m_rec.get_recommendations(userinput), topmovies=m_rec.topmoviesfn())
    return render_template("songs.html", title="Songs", songs="active", topsongs=s_rec.topsongsfn())

@app.route("/emotion",methods=['GET','POST'])
def emotion():
    return render_template("emotion.html", title="Emotion")

@app.route('/video_feed')
def video_feed():
    return Response(recommender.emotion.gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')    

@app.route("/team")
def team():
    return render_template("team.html", title="Team", team="active")

if __name__ == "__main__":
    app.run(debug=True)