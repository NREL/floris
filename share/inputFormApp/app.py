import json

from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    render_template,
    send_from_directory,
)


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config["SECRET_KEY"] = "a_super_secret_key"


@app.route("/")
def index():
    return send_file("templates/index.html")


@app.route("/generate/", methods=["POST"])
def generate_file():

    content = request.get_json(silent=True)
    data = {}
    if content:
        # parsing json data
        data = content
    else:
        try:
            data["data"] = json.loads(request.form["inputs"])
        except json.decoder.JSONDecodeError:
            resp = jsonify({"status": "error", "message": "invalid JSON"})
            resp.status_code = 500
            return resp
    try:
        output_directory = "./inputFiles"
        filename = data["name"]

        # write to file
        with open("{}/{}.json".format(output_directory, filename), "w") as outfile:
            json.dump(data, outfile, sort_keys=True, indent=2, separators=(",", ": "))

        resp = jsonify({"status": "success", "message": "file generated!"})

        resp.status_code = 200
        return resp

    except Exception as e:
        resp = jsonify({"status": "error", "message": "{}".format(e)})
        resp.status_code = 500
        return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0")
