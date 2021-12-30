from flask import Flask, flash, request, redirect, url_for
import json
import os
from flask_cors import CORS, cross_origin
import requests

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# global parameters for Voice Identification
data_audio = []
data_own = None
_rfc, _pca = [], []

@app.route("/health", methods=["GET"])
@cross_origin()
def health():
    return json.dumps({'status': 'OK'})


# route for testing upload audio
@app.route("/testUploadAudio", methods=["POST"])
@cross_origin()
def testUploadAudio():
    data = {"status": 'NotOK'}
    print(request.files)
    if request.files:
        audio = request.files["audio"]
        audio.save(os.path.join(app.config['UPLOAD_FOLDER'], 'audio.mp3'))
        print("Upload Audio Succesfully. Saved to upload folder")
        data["status"] = "OK"
    return json.dumps(data, ensure_ascii=False)



if __name__ == "__main__":
    print("App run!")
    app.run(debug=False, host="0.0.0.0", threaded=False)
