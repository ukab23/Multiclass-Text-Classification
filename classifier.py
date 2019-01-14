import nltk
import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from many_stop_words import get_stop_words
from nltk.tokenize import wordpunct_tokenize
from datetime import datetime
from gtts import gTTS
import pyglet

class Classifier():
    def __init__(self):
        # self.labels = list(filter(lambda x: not (x.endswith("~") or x.startswith("client")), os.listdir("./classifier/command_classifier/data/")))
        self.labels = [
                         "date.txt",
                         "day.txt",
                         "time.txt",

        ]
        self.weekdays = { 0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday' }
        self.vectorizer = TfidfVectorizer( max_features = None, strip_accents = 'unicode',
                            analyzer = "word", ngram_range=(1,1), use_idf = 1, smooth_idf = 1, stop_words='english')
        self.model_folder = "./classifier/command_classifier/models"
        self.model_path = self.model_folder+"/command_classifier.joblib"


        xdata = []
        ydata = []
        train_path ="./data/"
        for file in self.labels:
        	# print("File Exists")
        	with open(train_path+file,"r") as f:
        		data = f.readlines()
        		data = [i.replace("\n","") for i in data]
        		data = [i for i in data if i != ""]
        		for i in data:
        			xdata.append(i)
        			ydata.append(self.labels.index(file))
        vectors_train = self.vectorizer.fit_transform(xdata)
        self.classifier = OneVsRestClassifier( LogisticRegression( C = 10.0, multi_class = "multinomial", solver = "lbfgs" ) )
        self.classifier.fit(vectors_train, ydata)
        
    def test(self, query):
        classifier_result = {}
        vectors_test = self.vectorizer.transform([query])
        pred = self.classifier.predict(vectors_test)
        predp = self.classifier.predict_proba(vectors_test)
        predp = predp.tolist()[0]
        max_ind =  predp.index(max(predp))
        classifier_result["class"] = self.labels[max_ind][:-4]
        classifier_result["prediction"] = {self.labels[max_ind]: round(max(predp)*100.0, 2)*0.6}
        
        
        if(classifier_result["class"] == 'date'):
            print("Date is",datetime.date(datetime.now()))
            speak_date = datetime.date(datetime.now()).strftime('%m/%d/%Y')
            tts = gTTS(text='The date is' + speak_date, lang='en')
            tts.save("speak_date.mp3")
            music = pyglet.resource.media('speak_date.mp3')
            music.play()
            pyglet.app.run()
            
        if(classifier_result["class"] == 'day'):
            print("It is",self.weekdays[datetime.today().weekday()])
            speak_day = self.weekdays[datetime.today().weekday()]
            tts = gTTS(text='It is' + speak_day, lang='en')
            tts.save("speak_day.mp3")
            music = pyglet.resource.media('speak_day.mp3')
            music.play()
            pyglet.app.run()
            
        if(classifier_result["class"] == 'time'):
            print("Current time is",datetime.time(datetime.now()))
            speak_time = datetime.date(datetime.now())
            speak_time = speak_time.strftime('%I:%M')
            tts = gTTS(text='Current time is' + speak_time, lang='en')
            tts.save("speak_time.mp3")
            music = pyglet.resource.media('speak_time.mp3')
            music.play()
            pyglet.app.run()
        return classifier_result

if __name__ == '__main__':
        while(True):
            result = Classifier()
            qr = input("=>")
            result.test(qr)
