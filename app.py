import streamlit as st
import img_classification
import metrics

header = st.container()
model_prediction = st.container()
model_metrics = st.container()

if 'correct_prediction_count' not in st.session_state:
    st.session_state['correct_prediction_count'] = 0.0
if 'incorrect_prediction_count' not in st.session_state:
    st.session_state['incorrect_prediction_count'] = 0.0


with header:
    st.title("nofacebook")

with model_prediction:
    img_url = st.text_input("Enter image URL")
    if img_url:
        try:
            prepared_image = img_classification.url_to_image(img_url)
        except ValueError:
            st.write('Please enter a valid URL for an image.')
        classify = st.button("Detect if the Image Contains a Human")
        if classify:
            try:
                st.image(img_url)
                st.write("")
                st.write(img_classification.classify_img(prepared_image))
            except:
                pass
    correct_prediction = st.button('Correct')
    if correct_prediction:
        st.session_state['correct_prediction_count'] += 1.0
    incorrect_prediction = st.button('Incorrect')
    if incorrect_prediction:
        st.session_state['incorrect_prediction_count'] += 1.0

    else:
        st.write("Paste Image URL")


with model_metrics:
    st.title("Model Performance")
    st.subheader('User Reported Accuracy')
    try:
        st.write(st.session_state['correct_prediction_count'] / (st.session_state['correct_prediction_count'] +
                                                                 st.session_state['incorrect_prediction_count']), '%')
    except ZeroDivisionError:
        st.write('Awaiting user input')
    st.subheader('Accuracy vs Epochs')
    st.pyplot(fig=metrics.get_model_accuracy())
    st.subheader('Loss vs Epochs')
    st.pyplot(fig=metrics.get_model_loss())
    st.write('It appears that there is no correlation between an increase in accuracy or loss and an increase in epochs'
             ' that the model is trained on. I believe the model does not become more accurate as more epochs are '
             'done, because the model simply needs more data. This model was trained on only 729 images: 368 images '
             'containing a human and 361 images not containing a human. Another issue that may arise with this '
             'dataset is that the images were pulled from security cameras. The way a person looks in an image taken '
             'by another person and the way a person looks on a security camera are quite different and that will '
             'probably further affect the accuracy of this model')
