from argparse import ArgumentParser
from operator import itemgetter
import logging
import os
from urllib.parse import urlparse

from flask import Flask
from flask import current_app, request, jsonify, abort

import numpy as np

try: import tensorflow as tf
except ImportError: tf = None

# We use ycml full paths here so it is easy to copy and paste this code.
from ycml.utils import get_settings, load_dictionary_from_file, uri_open, URIFileType
from ycml.featclass import load_featclass
from ycml.http_daemon.decorators import check_api_token


def create_app(A, file_settings):
    log_level = get_settings(key='log_level', sources=('env', file_settings), default='INFO').upper()
    log_format = get_settings(key='log_format', sources=(A, 'env', file_settings), default='%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s')
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    config_dict = dict(
        DEBUG=log_level in ['DEBUG'],
        CSRF_ENABLED=True,
        SECRET_KEY=os.urandom(24),
    )
    config_dict['JSONIFY_PRETTYPRINT_REGULAR'] = config_dict['DEBUG']

    api_token = get_settings(key='http_daemon_api_token', sources=('env', file_settings))
    if api_token is None:
        http_daemon_uri = get_settings(key='http_daemon_uri', sources=('env', file_settings), raise_on_missing=True)
        api_token = os.path.basename(urlparse(http_daemon_uri).path)
    #end if
    config_dict['api_token'] = api_token

    app = Flask(__name__)  # create our flask app!
    app.config.update(config_dict)

    with app.app_context():
        current_app.config['featclass'] = load_featclass(settings=file_settings, uri=get_settings(key='featclass_uri', sources=('env', file_settings)))
        if tf: current_app.config['tf_graph'] = tf.get_default_graph()

        current_app.logger.setLevel(logging.getLevelName(log_level))
    #end with

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

        instances = o.get('instances', None)
        if instances is None:
            instances = [o]

        if not instances:
            current_app.logger.debug('Request does not have any instances!')
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
            else: Y, astype, epsilon = Y_predict, bool, -1.0

            if hasattr(model.classifier, 'unbinarize_labels'): unbinarized = model.classifier.unbinarize_labels(Y, to_dict=True, astype=astype, epsilon=epsilon)
            else: unbinarized = [dict((j, astype(Y[i, j])) for j in range(Y.shape[1]) if Y[i, j] > epsilon) for i in range(Y.shape[0])]

            if limit: results = [dict(sorted(unbinarized[i].items(), key=itemgetter(1), reverse=True)[:limit]) for i in range(Y_proba.shape[0])]
            else: results = list(unbinarized)

            return jsonify(results)

        except Exception:
            logging.exception('Exception while processing request from <{}>.'.format(remote_ip))
        #end try
    #end def

    return app
#end def


if __name__ == '__main__':
    parser = ArgumentParser(description='Starts the web daemon for URL classifier API.')
    parser.add_argument('settings_uri', type=URIFileType(), nargs='?', metavar='<settings_uri>', help='File containing daemon settings.')
    parser.add_argument('--settings', type=URIFileType(), metavar='<settings_uri>', help='File containing daemon settings.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')
    parser.add_argument('-p', '--port', type=int, default=None, metavar='p', help='Port to listen on.', dest='http_daemon_port')
    A = parser.parse_args()

    logging.getLogger('botocore').setLevel(logging.WARN)
    logging.getLogger('boto3').setLevel(logging.WARN)
    logging.getLogger('requests').setLevel(logging.WARN)

    if A.settings: settings_uri = A.settings
    else: settings_uri = get_settings(key='settings_uri', sources=('env', A), raise_on_missing=True)

    if isinstance(settings_uri, str): settings_uri = uri_open(settings_uri)
    file_settings = load_dictionary_from_file(settings_uri)
    settings_uri.close()

    http_daemon_port = get_settings(key='http_daemon_port', sources=('env', A, file_settings))
    if http_daemon_port is None:
        http_daemon_uri = get_settings(key='http_daemon_uri', sources=('env', file_settings), raise_on_missing=True)
        http_daemon_port = urlparse(http_daemon_uri).port
    #end if

    if http_daemon_port is None:
        http_daemon_port = 5000
    http_daemon_port = int(http_daemon_port)

    app = create_app(A, file_settings)
    app.run(debug=A.debug, host='0.0.0.0', use_reloader=False, port=http_daemon_port)
#end if
