# -*- coding:utf-8 -*-
from flask import Flask, render_template, request
from config import config
from predict import Predict


app = Flask(__name__)
predictor = Predict(config)
print('data prepare finished')


@app.route('/comment', methods=['GET', 'POST'], strict_slashes=False)
def extract_views():
    print('get request')
    comment = request.args.get('foodComment')
    if not comment or len(comment.strip()) == 0:
        comment_class = []
    else:
        comment_class = predictor.predict(comment)
        # print(views)
    print(comment_class)
    return render_template('index.html', comment_class=comment_class)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9998')
