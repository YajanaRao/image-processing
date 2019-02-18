from flask import Flask, jsonify, Response
from Camera import Camera
from flask_cors import CORS
from processors import img_processor


app = Flask(__name__)
CORS(app)


@app.route('/video_feed')
def video_feed():
    return Response(feed(Camera()),	mimetype='multipart/x-mixed-replace; boundary=frame')


def feed(camera):
    while True:
        try:
            frame = camera.get_frame()
            frame = img_processor(frame)
            # text_processor(frame)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as exp:
            return exp

if __name__ == "__main__":
    app.run(threaded=True)
