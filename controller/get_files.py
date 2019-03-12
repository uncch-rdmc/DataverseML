from flask import Blueprint, render_template, session, abort, request, jsonify, redirect
import requests
from key import *

app_get_files = Blueprint('app_get_files', __name__)

#############################################################################
# /get_files
# Get file names from Dataverse
#############################################################################
@app_get_files.route("/get_files", methods=['GET','POST'])
def get_files():
	if request.args.get("doi"):
		global API_KEY
		doi = request.args.get("doi")
		r = requests.get("https://dataverse-test.irss.unc.edu/api/datasets/:persistentId/?persistentId=doi:"+doi+"&key="+API_KEY)
		return r.text