from argparse import ArgumentParser
import json
import logging
from operator import itemgetter
import os
from urllib.parse import urlparse

from flask import Flask
from flask import current_app, request, jsonify, abort

import numpy as np

try: import tensorflow as tf
except ImportError: tf = None

from uriutils import URIFileType

from ycsettings import Settings

# We use ycml full paths here so it is easy to copy and paste this code.
from ycml.featclass import load_featclass
from ycml.http_daemon.decorators import check_api_token

logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger = logging.getLogger('ycml.http_daemon')


def create_app(A, settings):
    log_level = settings.get('log_level', default='INFO').upper()
    log_format = settings.get('log_format', default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    config_dict = dict(
        DEBUG=log_level in ['DEBUG'],
        CSRF_ENABLED=True,
        SECRET_KEY=os.urandom(24),
    )
    config_dict['JSONIFY_PRETTYPRINT_REGULAR'] = config_dict['DEBUG']

    api_token = settings.get('http_daemon_api_token')
    if api_token is None:
        http_daemon_uri = settings.get('http_daemon_uri', raise_exception=True)
        api_token = os.path.basename(urlparse(http_daemon_uri).path)
    #end if
    config_dict['api_token'] = api_token

    app = Flask('ycml.http_daemon')  # create our flask app!
    app.config.update(config_dict)

    with app.app_context():
        current_app.config['featclass'] = load_featclass(settings=settings, uri=settings.get('featclass_uri'))
        if tf: current_app.config['tf_graph'] = tf.get_default_graph()
    #end with

    @app.route('/', methods=['GET', 'POST'])
    def index():
        return 'Toto, I\'ve a feeling we\'re not in Kansas anymore.', 404

    @app.route('/api/<api_token>/ping', methods=['GET'])
    @check_api_token
    def api_ping():
        return jsonify(dict(success=True))
    #end def

    @app.route('/api/<api_token>', methods=['POST'])
    @check_api_token
    def api():
        o = request.get_json(force=True, silent=True)
        if o is None: abort(400)

        try: remote_ip = request.headers.getlist('X-Forwarded-For')[0]
        except Exception: remote_ip = None

        instances = o.get('instances')
        if instances is None:
            instances = [o]

        if not instances:
            logger.debug('Request does not have any instances!')
            return jsonify(dict(reason='Request does not have any instances!')), 400
        #end if

        params = o.get('params', {})

        try:
            model = current_app.config['featclass']
            X = np.array(instances, dtype=np.object)

            if tf:
                with current_app.config['tf_graph'].as_default():
                    Y_proba, Y_predict = model.predict_and_proba(X, **params)
            else:
                Y_proba, Y_predict = model.predict_and_proba(X, **params)
            #end with

            limit = o.get('limit', False)
            predicted_only = o.get('predicted_only', False)
            probabilities = o.get('probabilities', False)

            if predicted_only:
                Y_proba = np.multiply(Y_proba, Y_predict)  # zeros out non predicted values

            if probabilities: Y, astype, epsilon = Y_proba, float, 0.0
            else: Y, astype, epsilon = Y_predict, bool, 0.0

            if hasattr(model.classifier, 'unbinarize_labels'): unbinarized = model.classifier.unbinarize_labels(Y, to_dict=True, astype=astype, epsilon=epsilon)
            else: unbinarized = [dict((j, astype(Y[i, j])) for j in range(Y.shape[1]) if Y[i, j] > epsilon) for i in range(Y.shape[0])]

            if limit: results = [dict(sorted(unbinarized[i].items(), key=itemgetter(1), reverse=True)[:limit]) for i in range(Y_proba.shape[0])]
            else: results = list(unbinarized)

            o['instances'] = '<{} instances>'.format(len(instances))
            logger.debug('Request={}; Response={}'.format(json.dumps(o), json.dumps(results)))

            return jsonify(results)

        except Exception:
            logging.exception('Exception while processing request from <{}>.'.format(remote_ip))
        #end try
    #end def

    return app
#end def


def gunicorn_app(environ, start_response):
    app = create_app({}, Settings())

    return app(environ, start_response)
#end def


if __name__ == '__main__':
    parser = ArgumentParser(description='Starts the web daemon for URL classifier API.')
    parser.add_argument('settings_uri', type=URIFileType(), nargs='?', metavar='<settings_uri>', help='File containing daemon settings.')
    parser.add_argument('--settings', type=URIFileType(), metavar='<settings_uri>', help='File containing daemon settings.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')
    parser.add_argument('-p', '--port', type=int, default=None, metavar='p', help='Port to listen on.', dest='http_daemon_port')
    A = parser.parse_args()

    settings = Settings(A)

    http_daemon_port = settings.get('http_daemon_port')
    if http_daemon_port is None:
        http_daemon_uri = settings.get('http_daemon_uri', raise_exception=True)
        http_daemon_port = urlparse(http_daemon_uri).port
    #end if

    if http_daemon_port is None:
        http_daemon_port = 5000

    http_daemon_port = int(http_daemon_port)

    app = create_app(A, settings)
    app.run(debug=A.debug, host='0.0.0.0', use_reloader=False, port=http_daemon_port)
#end if
