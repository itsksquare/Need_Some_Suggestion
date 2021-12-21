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
        userminput = request.form['movie_name']
        return render_template("movies1.html", title="Movies Recommender", movies="active", recmovies=m_rec.get_recommendations(userminput), topmovies=m_rec.topmoviesfn())
    return render_template("movies.html", title="Movies Recommender", movies="active", topmovies=m_rec.topmoviesfn())

@app.route("/songs", methods=['POST', 'GET'])
def songs():
    if flask.request.method == 'POST':
        usersinput = request.form['song_name']
        return render_template("songs1.html", title="Songs Recommender", songs="active", recsongs=s_rec.s_recommend(usersinput), topsongs=s_rec.topsongsfn())
    return render_template("songs.html", title="Songs Recommender", songs="active", topsongs=s_rec.topsongsfn())

@app.route("/emotion",methods=['GET','POST'])
def emotion():
    return render_template("emotion.html", title="Emotion", recs=m_rec.emotionmovie(), recm=s_rec.emotionsong())

@app.route('/video_feed')
def video_feed():
    return Response(recommender.emotion.gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')    

@app.route("/team")
def team():
    return render_template("team.html", title="Team", team="active")

if __name__ == "__main__":
    app.run(debug=True)