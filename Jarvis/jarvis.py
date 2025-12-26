import pyttsx3 #pip install pyttsx3
import datetime
import speech_recognition as sr #pip install SpeechRecognition
import wikipedia #pip install wikipedia
import smtplib #email
import webbrowser as wb #search Google
import os #inbuilt library for the operating system
import pyautogui #pip install pyautogui
import psutil #pip install psutil
import pyjokes #pip install pyjokes


engine = pyttsx3.init()

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1 #wait for one second 
        audio = r.listen(source) #collect audio from source
        
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(query)

    except Exception as e:
        print(e)
        speak("Kindly Repeat yourself, Sir H!")
        
        return "None"
    return query
# Greet Jarvis 
takeCommand()

def greet():
    speak("Welcome Back, Helmer!!")
    hour = datetime.datetime.now().hour
    if hour >= 6 and hour < 12:
        speak("Good morning, Sir H!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon, Sir H!")
    elif hour >= 18 and hour < 24:
        speak("Good Evening, Sir H!")
    else:
        speak("Let's get to work, Sir H!")
# greet()

def time():
    Time = datetime.datetime.now().strftime("%I:%M:%S")
    speak("The current time is")
    speak(Time)
# time()

def date():
    year = int(datetime.datetime.now().year)
    month = int(datetime.datetime.now().month)
    day = int(datetime.datetime.now().day)
    speak("Today's Date is")
    speak(day)
    speak(month)
    speak(year)
# date()

def dailyUpdate():
    speak("The Daily Update, Sir H")
    time()
    date()
# dailyUpdate()

def beckon():
    speak("EDITH is at your service, How may I be of Assistance? ")
# beckon()

def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('michaeleriayo0@gmail.com', 'password') #enter your password here and email to preferred mail of your choice
    server.sendmail('michaeleriayo0@gmail.com', to, content)
    server.close()

def cpu():
    usage = str(psutil.cpu_percent())
    speak("CPU is at" + usage)
    battery = psutil.sensors_battery()
    speak("Your Battery is at" + battery.percent)
    
def joke():
    speak(pyjokes.get_joke())
    
if __name__ == "__main__":
    greet()
    while True:
        query = takeCommand().lower()
        
        if 'time' in query:
            time()
        elif 'date' in query:
            date()
        elif 'hello' in query:
            greet()
        elif 'daily update' in query:
            dailyUpdate()
        elif 'wikipedia' in query:
            speak("Searching Wikipedia...") # search on wikipedia, "....."
            query = query.replace("wikipedia", "")
            result = wikipedia.summary(query, sentences=2)
            print(result)
            speak(result)
        elif 'send email' in query:
            try:
                speak("What message do you want to send?")
                content = takeCommand()
                to = 'temimichael27@gmail.com'
                speak(content)
                sendEmail(to, content)
                speak("Your Email has been successfully sent")
            except Exception as e:
                print(e)
                speak("Unable to send email, please try again")
            speak("What should I search for?") # what you want to search for
            chromePath = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Google Chrome.lnk"
            search = takeCommand().lower()
            wb.get(chromePath).open_new_tab(search + '.com') # .com if it is a website, if search remove
        elif 'logout' in query:
            os.system("shutdown -l") #lock our system
        # elif 'restart' in query:
        #     os.system("shutdown /r /t 1") #restart our system
        # elif 'shutdown' in query:
        #     os.system("shutdown /s /t 1") #shutdown our system
        # elif 'play song' in query:
        #     song_file_path = "C:\Users\MICHEAL\Desktop\gmt\songs"
        #     songs = os.listdir(song_file_path)
        #     speak("YO YO!! It's DJ X, you know I got you. Now, Let's Start off with something nice!")
        #     os.startfile(os.path.join(song_file_path, songs[0]))
        elif 'take note' in query: #Jarvis, take note of...
            speak("What should I take note of, Helmer")
            data = takeCommand()
            speak("Taking note of..." + data)
            remember = open('jarvisData.txt','w')
            remember.write(data)
            speak("Note Stored")
            remember.close()
        elif 'what do you know' in query: #ask Jarvis what he knows
            remember = open('jarvisData.txt', 'r')
            speak("You once said" + remember.read())
        elif 'cpu' in query:
            cpu()
        elif 'Matthew' in query:
            speak("reading")
            speak("For the gate is narrow and the way is hard that leads to life, and those who find it are few.")
        elif 'joke' in query:
            speak(pyjokes.get_joke())
            # or joke()
       
            
        elif 'offline' in query: # go offline, EDITH
            quit()
            
                   
            
                
                
                
                
                
                
                
                
                
# smart room tools, IOT tools, pi board, Esp ..266
                
# activate less secure apps in your gmail
# learn 'try', 'except', 'if __main__', 'except Exception', 'server.ehlo.starttls
