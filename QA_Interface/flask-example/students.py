# using python 3
from flask import Flask, render_template
app = Flask(__name__)

# two decorators using the same function
@app.route('/')
@app.route('/index.html')
def index():
    return '<h1>Welcome to the student records Flask example!<h1>'

# your code here


# keep this as is
if __name__ == '__main__':
    app.run(debug=True, port=3134)
