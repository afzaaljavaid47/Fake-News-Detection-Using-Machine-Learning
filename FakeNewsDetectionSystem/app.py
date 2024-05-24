import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
pre_trained_model = pickle.load(open('preTrainedModel', 'rb'))
serialized_news_data_object = pickle.load(open('serializeDataObject','rb'))

@app.route('/')
def home():
	return render_template('Index.html')

@app.route('/predict_news',methods=['POST'])
def predict_news():
    if request.method == 'POST':
    	news_article_data = request.form['news']
    	news_article_data = [news_article_data]
    	serialized_news_data = serialized_news_data_object.transform(news_article_data)
    	predicted_output = pre_trained_model.predict(serialized_news_data)
    	return render_template('Output.html', output=predicted_output)

if __name__ == '__main__':
    app.run(debug=True)
