from flask import Blueprint, render_template, session, abort, request, jsonify, redirect

app_index = Blueprint('app_index', __name__)

#############################################################################
# /index
# Main Page of the course
#############################################################################
@app_index.route("/", methods=['GET','POST'])
@app_index.route("/index", methods=['GET','POST'])
def index():
	return render_template("index.html")

    
	
	
