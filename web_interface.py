# web_interface.py
# Creates html file to stream video feed to web browser.

import flask, cv2, datetime, threading, argparse
from MotionDetector import MotionDetector

# Global variables
app = flask.Flask(__name__.split('.')[0])
videoFeed = cv2.VideoCapture(0)
outputFrame = None
frameLock = threading.Lock()

@app.route("/")
def index():
    # Return the rendered html file
    return flask.render_template("index.html")


def detect_motion(frameCount):
    global videoFeed, outputFrame, frameLock

    md = MotionDetector()
    total_frames = 0

    while True:
        # Capture frame
        ret, frame = videoFeed.read()

        # Format image into gray scale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale image
        gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

        # Timestamp the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if total_frames > frameCount:
            # Detect motion
            motion = md.detect(gray_image)
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (minX, minY, maxX, maxY) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
        else:
            total_frames += 1
        
        # Update the background model
        md.update(gray_image)

        # Acquire the lock, update the output frame, then release the lock
        with frameLock:
            outputFrame = frame.copy()

        # @NOTE: may cause issue if running in thread
        # Cleanup and exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            videoFeed.release()
            cv2.destroyAllWindows()
            break


# Generate a jpg file of the image and format into a byte array
def generate():
    # Grab global references to the output frame and frame lock variables
    global outputFrame, frameLock

    # Loop over frames from the output stream
    while True:
        # Wait until the lock is acquired
        with lock:
            if outputFrame is None:
                continue

            # Encode the frame into jpg format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

        # @NOTE make sure this is correct rv logic
        # Ensure the frame was successfully encoded
        if not flag:
            continue

        # Yield a byte array of the output frame
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # Return the generated response and media type
    return flask.Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


# Begin program
if __name__ == '__main__':
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # Start a thread that will perform motion detection
    detection_thread = threading.Thread(target=detect_motion, args=(args["frame_count"],))
    #t.daemon = True # NOTE try without
    detection_thread.start()

    # Start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)

