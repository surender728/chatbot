# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:40:16 2020

@author: aa
"""


from flask import Flask, render_template, request
import tensorflow as tf
import build_chatbot as bot
import create_model as cm




app = Flask(__name__)
# model  = cm.build_model()
# model.load_weights('chatBotModel')
# model._make_predict_function()
# graph = tf.get_default_graph()

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/get")
def get_bot_response():
    # import pdb;pdb.set_trace()
    # global graph
    # global model
    # with graph.as_default():
    userText = request.args.get('msg')
    print(userText)
    return str(bot.response(userText))

if __name__ == "__main__":
    app.run()