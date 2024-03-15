#-----------------------------------Importing libraries
from flask import Flask, jsonify, request ,render_template

from data_preprocessing import remove_spaces,expand_text,handling_accented,clean_data,lemmatization,join_list


import pickle
#-----------------------------------Importing libraries

#-----------------------------------Initialization
app = Flask(__name__)
#-----------------------------------Initialization

#-----------------------------------Config data
tfidf_model = pickle.load(open('models/tfidf.pkl', 'rb'))

model = pickle.load(open('models/model.pkl', 'rb'))



#-----------------------------------test route
@app.route('/')
def home():
    return jsonify({'response' : 'This is home !'})
#-----------------------------------test route

#-----------------------------------prediction route
@app.route('/predict', methods = (['GET','POST']))
def analyze_sentiment():

    if request.method == "POST":

        requested_data = request.get_data(as_text = True)
        
        clean_text_test = remove_spaces(requested_data)
        
        clean_text_test = expand_text(clean_text_test)
        
        clean_text_test = handling_accented(clean_text_test)
        
        clean_text_test = clean_data(clean_text_test)
        
        clean_text_test = lemmatization(clean_text_test)
        
        clean_text_test = join_list(clean_text_test)
        
        vector = tfidf_model.transform([clean_text_test])
        
        result = model.predict(vector)
        # if sentiment[0]==0:
        #     result = "Positive Sentiment"
        # elif sentiment[0]==1:
        #     result = "Ngative Sentiment"
        # elif sentiment[0]==2:
        #     result = "Positive Sentiment"

        
        return render_template('index.html', sentiment=result)
    
    return render_template('index.html')

#-----------------------------------run the app
if __name__ == '__main__':
    app.run(debug=True)
#-----------------------------------run the appp