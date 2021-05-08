
from random import random
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('Agg')
from io import BytesIO
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    # img in the html specifies the presence of an image
    # however, the src of the image is a routing block instead of
    # a .jpg or .png
    return '''
        <h2>Here's my plot!</h2>
        <img src='/plot.png'>
        '''

@app.route('/plot.png')
def get_graph():
    fig = Figure()
    ax = fig.subplots()
    n = 10
    ax.plot(range(n), [random() for i in range(n)])
    image = BytesIO()
    fig.savefig(image)  #the plot is saved
    return image.getvalue(), 200, {'Content-Type': 'image/png'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
