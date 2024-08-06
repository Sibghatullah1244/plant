import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_set.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset

                The plant disease detection dataset comprises approximately 17,000 high-resolution images of various plant species, each labeled with information on the presence or absence of specific diseases. This diverse dataset aims to aid in developing automated systems for early and accurate disease detection.
                 By using this dataset, farmers can leverage advanced AI tools to identify diseases at an early stage, preventing the spread and reducing crop loss. Accurate disease detection helps in timely intervention, improving overall crop health and yield. Additionally, it reduces the reliance on expert knowledge, making advanced diagnostics accessible to a broader farming community.
                #### Content
                1. train (17112 images)
                2. test (33 images)
                3. validation (4278 images)

                """)

#Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=400, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name = ['Apple___Apple_scab', 'Apple___Cedar_apple_rust', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 
                      'Potato___Early_blight', 'Potato___healthy',  
                      'Tomato___Early_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                      'Tomato___healthy']
        
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
