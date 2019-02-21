from flask import Flask, jsonify, Response, render_template
from Camera import Camera
from flask_cors import CORS
from processors import img_processor, text_detector, emotion_detector
import threading
import cv2

app = Flask(__name__)
CORS(app)

frame = None


@app.route('/text')
def text_feed():
    try:
        if frame.any():
            text = text_detector(frame)
    except Exception as exp:
        print(exp)
        text = "no text"
        
    return jsonify(text)

@app.route('/emotion')
def emotion_feed():
    try:
        emotion_detector(frame)
    except Exception as exp:
        print(exp)
        return jsonify("error")

@app.route('/video_feed')
def video_feed():
    return Response(feed(Camera()),	mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


def feed(camera):
    while True:
        try:
            global frame
            frame = camera.get_frame()
            img = img_processor(frame)
            if img.any():
                frame = img
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        except Exception as exp:
            return exp

if __name__ == "__main__":
    app.run(threaded=True)
