from flask import Flask, render_template, json, request, session

##############################
# app initialization
###############################
app = Flask(__name__)
app.secret_key = "b'\\xb4\\xde\\xd9\\x86\\x93\\xb8\\x1bg\\x93@8W\\xc2&Dn4\\xf4\\xd0\\xa6\\x92\\x13XO'" 
###randomizing will clear flask_login session when app is restarted
###app.secret_key = str(os.urandom(24)) 

##############################
# controller initialization
###############################
from controller.index import app_index
from controller.get_files import app_get_files
from controller.analyze_data import app_analyze_data

# from controller.testwatson import app_testwatson

app.register_blueprint(app_index)
app.register_blueprint(app_get_files)
app.register_blueprint(app_analyze_data)

if __name__ == "__main__":
    app.run(debug=True, port=80)