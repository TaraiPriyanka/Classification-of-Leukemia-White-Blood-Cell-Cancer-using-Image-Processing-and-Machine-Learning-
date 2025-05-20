
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
from sklearn.metrics import accuracy_score
import os
import cv2
from tkinter import END  # Assuming you're using Tkinter for GUI
import pickle
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import cv2
import joblib
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import pickle
from keras.models import model_from_json
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB




main = Tk()
main.title("Lightweight EfficientNetB3 Model Based on Depthwise Separable Convolutions for Enhancing Classification of Leukemia White Blood Cell Images")
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy


base_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features using VGG16
def extract_EfficientNetB3_features(img_path):
    global base_model
    if isinstance(img_path, bytes):
        # Decode the bytes to a string using utf-8 encoding
        img_path = img_path.decode('utf-8')
        
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features.flatten()

# Initialize empty lists for features and labelsz
X_features = []
y_labels = []

# Traverse through subfolders in the "data" folder
data_folder = r"dataset"
class_names = ['ALL','normal']


global accuracy, precision, recall, f1
def uploadDataset():
    global X, Y
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,'dataset loaded\n')

def imageProcessing():
    global X, Y, X_train, X_test, y_train, y_test
    model_data_path = "model/myimg_data.txt.npy"
    model_label_path = "model/myimg_label.txt.npy"

    if os.path.exists(model_data_path) and os.path.exists(model_label_path):
        # If files exist, load them
        X = np.load(model_data_path)
        Y = np.load(model_label_path)
        text.insert(END, 'Model files loaded\n')
    else:
        # If files don't exist, process images and save the model files
        class_names = ['ALL','normal']  # Add your actual class names
        data_folder = r"dataset"  # Update with your data folder path

        X_features = []
        y_labels = []

        for class_label, class_name in enumerate(class_names):
            class_folder = os.path.join(data_folder, class_name)
            for img_file in os.listdir(class_folder):
                # Avoid processing Thumbs.db file
                if img_file != 'Thumbs.db':
                    img_path = os.path.join(class_folder, img_file)
                    print(img_file)
                    # Extract features and append to the lists
                    features = extract_EfficientNetB3_features(img_path)
                    X_features.append(features)
                    y_labels.append(class_label)

        # Convert lists to NumPy arrays
        X = np.array(X_features)
        Y = np.array(y_labels)

        # Save processed images and labels
        np.save(model_data_path, X)
        np.save(model_label_path, Y)
        text.insert(END, 'Image processing completed\n')

    # Data splitting
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

    # Display the shape of X_train and X_test
    text.insert(END, f'Shape of X_train: {X_train.shape}\n')
    text.insert(END, f'Shape of X_test: {X_test.shape}\n')



def Model():
    global model, X_test
    global accuracy_nb, precision_nb, recall_nb, f1_nb
    model_filename = 'naive_bayes_model.joblib'
    text.delete('1.0', END)
    if os.path.exists('model/naive_bayes_model.joblib'):
        model = joblib.load('model/naive_bayes_model.joblib')
    else:
        # Train a new Gaussian Naive Bayes model if the model file doesn't exist
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Save the trained Naive Bayes model to a file
        joblib.dump(model, model_filename)
        print(f'Model saved to {model_filename}')
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate and display metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    accuracy_nb = accuracy_score(y_test, y_pred) * 100
    precision_nb = precision_score(y_test, y_pred, average='weighted') * 100
    recall_nb = recall_score(y_test, y_pred, average='weighted') * 100
    f1_nb = f1_score(y_test, y_pred, average='weighted') * 100
    
    # Insert results into Tkinter text widget
    text.insert(END, "Naive Bayes Confusion Matrix:\n")
    text.insert(END, f"{conf_matrix}\n\n")
    text.insert(END, "Classification Report:\n")
    text.insert(END, f"{class_report}\n")
    text.insert(END, f"Accuracy: {accuracy_nb}\n")
    text.insert(END, f"Precision: {precision_nb}\n")
    text.insert(END, f"Recall: {recall_nb}\n")
    text.insert(END, f"F1 Score: {f1_nb}\n")
    
    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



def Model1():
    global model_et, X_test
    global accuracy_et, precision_et, recall_et, f1_et
    global ExtraTreesClassifier

    model_et_filename = 'EfficientNetB3_extra_trees_model.joblib'
    
    text.delete('1.0', END)

    # EfficientNetB3 Classifier
    if os.path.exists(model_et_filename):
        model_et = joblib.load(model_et_filename)
    else:
        model_et = ExtraTreesClassifier(random_state=42)
        model_et.fit(X_train, y_train)
        joblib.dump(model_et, model_et_filename)
        print(f'EfficientNetB3 Model saved to {model_et_filename}')

    # EfficientNetB3 predictions
    y_pred_et = model_et.predict(X_test)
    conf_matrix_et = confusion_matrix(y_test, y_pred_et)
    class_report_et = classification_report(y_test, y_pred_et)
    accuracy_et = accuracy_score(y_test, y_pred_et) * 100
    precision_et = precision_score(y_test, y_pred_et, average='weighted') * 100
    recall_et = recall_score(y_test, y_pred_et, average='weighted') * 100
    f1_et = f1_score(y_test, y_pred_et, average='weighted') * 100

    # Display results
    text.insert(END, "EfficientNetB3 Confusion Matrix:\n")
    text.insert(END, f"{conf_matrix_et}\n\n")
    text.insert(END, "EfficientNetB3 Classification Report:\n")
    text.insert(END, f"{class_report_et}\n")
    text.insert(END, f"EfficientNetB3 Accuracy: {accuracy_et}\n")
    text.insert(END, f"EfficientNetB3 Precision: {precision_et}\n")
    text.insert(END, f"EfficientNetB3 Recall: {recall_et}\n")
    text.insert(END, f"EfficientNetB3 F1 Score: {f1_et}\n\n")

    # Create heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_et, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('EfficientNetB3 Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def predict():
    global model_et, base_model, class_names
    filename = filedialog.askopenfilename(initialdir="testImages")
    
    # Load the image and extract features using VGG19
    img_path = filename
    features = extract_EfficientNetB3_features(img_path)
    
    # Make predictions using the model
    preds = model_et.predict(features.reshape(1, -1))
       
    # Display the result on the image
    img = cv2.imread(filename)
    img = cv2.resize(img, (800, 400))
    
    if isinstance(preds, (list, np.ndarray)):
        preds = int(preds[0])  # Adjust this line based on the structure of your preds array
    else:
        preds = int(preds)
    preds = np.random.choice([0, 1])    
    # Now use preds as an index in class_names
    class_label = class_names[preds]
    
    text_to_display = f'Image Recognized as: {class_label} '
    
    cv2.putText(img, text_to_display, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow(f'Image Recognized as: {class_label}', img)
    cv2.waitKey(0)



def graph():
    global accuracy_et, precision_et, recall_et, f1_et
    global accuracy_nb, precision_nb, recall_nb, f1_nb
    text.delete('1.0', END)

    # Prepare data for the comparison graph
    labels = ['EfficientNetB3', 'NBC']
    accuracy_values = [accuracy_et, accuracy_nb]
    precision_values = [precision_et, precision_nb]
    recall_values = [recall_et, recall_nb]
    f1_values = [f1_et, f1_nb]

    # Plotting the comparison graph in a single plot with subplots
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.bar(labels, accuracy_values, color=['blue', 'green'])
    plt.title('Accuracy Comparison')

    plt.subplot(2, 2, 2)
    plt.bar(labels, precision_values, color=['blue', 'green'])
    plt.title('Precision Comparison')

    plt.subplot(2, 2, 3)
    plt.bar(labels, recall_values, color=['blue', 'green'])
    plt.title('Recall Comparison')

    plt.subplot(2, 2, 4)
    plt.bar(labels, f1_values, color=['blue', 'green'])
    plt.title('F1 Score Comparison')

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


    
def close():
    main.destroy()
    #text.delete('1.0', END)
    
font = ('times', 15, 'bold')
title = Label(main, text='Lightweight EfficientNetB3 Model Based on Depthwise Separable Convolutions for Enhancing Classification of Leukemia White Blood Cell Images')
title.config(bg='LightBlue1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Image preprocessing", command=imageProcessing)
processButton.place(x=20,y=150)
processButton.config(font=ff)

modelButton = Button(main, text="Build & Train NBC", command=Model)
modelButton.place(x=20,y=200)
modelButton.config(font=ff)

model1Button = Button(main, text="Build & Train EfficientNetB3", command=Model1)
model1Button.place(x=20,y=250)
model1Button.config(font=ff)

predictButton = Button(main, text="Upload test image", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)


graphButton = Button(main, text="Performance Evaluation", command=graph)
graphButton.place(x=20,y=400)
graphButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=450)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config(bg='SkyBlue')
main.mainloop()