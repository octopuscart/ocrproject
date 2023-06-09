#!/usr/bin/python3.6

import sys
import site

#site.addsitedir('/var/www/hitme/lib/python3.6/site-packages')

sys.path.insert(0, '/home/varbin/ocr.varbin.com')

from app import app as application
#from gevent.pywsgi import WSGIServer


#http_server = WSGIServer(('', 8020), app)
#http_server.serve_forever()
