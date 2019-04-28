import sys

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin

from .my_exceptions import NotInCorpus
from .word2vec.reddit_model import RedditModel, NotInCorpusError

reddit_model = RedditModel()
app = Flask(__name__)
CORS(app)


@app.route('/similar')
def similar():
    search_term = request.args.get('searchTerm')
    n = int(request.args.get('n'))
    error = None
    words = []
    
    try:
        words = reddit_model.get_nearest(search_term, n)
    except NotInCorpusError as e:
        
        raise NotInCorpus("Words {} not in corpus".format(search_term), status_code=404)

    return jsonify({
        'searchTerm': search_term,
        'words': words
    })

@app.route('/similarity')
def similarity():
    word1 = request.args.get('word1')
    word2 = request.args.get('word2')

    try:
        similarity = reddit_model.get_similarity(word1, word2)
    except NotInCorpusError as e:
        msg = 'At least on of the searched words ({}, {}) not found in corpus'.format(word1, word2)
        raise NotInCorpus(msg, status_code=404)

    return jsonify({'similarity': str(similarity)})

@app.route('/similar2')
def similarity2():
    p1 = request.args.get('p1')
    p2 = request.args.get('p2')
    m1 = request.args.get('m1')
    n = int(request.args.get('n'))
    
    try:
        words = reddit_model.get_nearest_algebra(positive=[p1, p2], negative=[m1], n=n)
    except NotInCorpusError as e:
        msg = 'At least on of the searched words not found in corpus'
        raise NotInCorpus(msg, status_code=404)

    return jsonify({'words': words})



@app.errorhandler(NotInCorpus)
def handle_not_in_corpus(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
