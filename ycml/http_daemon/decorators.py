from functools import wraps

from flask import current_app, abort

__all__ = ['check_api_token']


def check_api_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_token = kwargs.pop('api_token')
        if api_token != current_app.config['api_token']: abort(403)

        return f(*args, **kwargs)
    #end def

    return decorated_function
#end def
