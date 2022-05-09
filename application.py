
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
from Camera import Camera

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
global feed_status
global capture_status
feed_status = {}
capture_status = "not specified"

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/sign_in")
def signin():
    return render_template("signin.html", name="notspecified")

@app.route("/sign_up", methods=['POST'])
def signup():
	name = request.form.get("uname")
	print(name)
	return render_template("signup.html", name=name)


def gen(camera, person):
	global feed_status
	feed_status[person] = 0
	print(feed_status)
	while True:
		#print("going to get frame")
		status, frame = camera.getframe() #the difference between signup and signin is the obj provided
		#print(status)
		yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')
		if status == 1:
			print(status)
			feed_status[person] = 1			
			print('setting up status to 1')
			break


@app.route('/sign_up/api')
def image_feed():
	name = request.args.get("name")
	cam = Camera(name)
	fobject = gen(cam, person=name)
	return Response(fobject,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign_up/status', methods=['POST'])
def image_feed_status():
	if request.method == "POST":
		json_data = request.get_json()
	name = json_data[0]['msg']
	print("jason name:",name)
	print(feed_status)
	if name in feed_status:
		status = feed_status[name]
	else:
		status = 0
	result = {'message' : status}
	return jsonify(result)


@app.route('/sign_in/api')
def image_capture():
	print("Inside here")
	cam = Camera("notspecified")
	fobject = gendetect(cam, person="notspecified")
	print("here")
	return Response(fobject,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gendetect(camera, person):
	print("gendetect")
	global capture_status
	while True:
		status, frame, ID, name = camera.detectface() #the difference between signup and signin is the obj provided
		print(status)
		yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')
		if status == 1:
			capture_status = name
			print("found a person",name)
			break

@app.route('/sign_in/status', methods=['POST'])
def img_capture_status():
	if request.method == "POST":
		json_data = request.get_json()
	name = json_data[0]['msg']
	print("jason name:",name)
	print(capture_status)
	# if name in feed_status:
	# 	status = feed_status[name]
	# else:
	# 	status = 0
	result = {'message' : capture_status}
	return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)