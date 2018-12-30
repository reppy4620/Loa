from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from network import UNet, NormType

import os
import processing
import torch


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'JPG'}

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
net = UNet(norm=NormType.Batch).to(device)
net.load_state_dict(torch.load('D:/Pix2Pix/Models/best.pth')['netG'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    mode = request.form.getlist("md")[0]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if mode == 'l2c':
            processing.line2color(filename, net, device)
        elif mode == 'c2l':
            processing.color_to_line(filename)
        else:
            raise ValueError("mode is invalid value.")
        return redirect(url_for('uploaded_file',
                                filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.debug = True
    app.run()
