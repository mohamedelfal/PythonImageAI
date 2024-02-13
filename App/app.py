from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    description = request.json['description']
    # هنا يمكنك كتابة الكود لتوليد الصورة بناءً على الوصف
    # في هذا المثال، سنقوم بإرجاع عنوان URL لصورة عشوائية
    image_url = "https://picsum.photos/500/500"
    return jsonify({"image_url": image_url})

if __name__ == '__main__':
    app.run(debug=True)
