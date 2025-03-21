import os
import streamlit as st
from PIL import Image
from leaf_utils import LeafImageProcessor
from DNA_Models import DNA_Models
import cv2
import numpy as np

css = '''<style>
@media (min-width: 1076px) {.st-emotion-cache-1y4p8pa {padding-left: 1rem; padding-right: 1rem;}} 
        .title {font-size: 26px; font-weight: bold; color: #4CAF50; /* Green color */ text-align: center; margin-top: 20px; }
        .st-emotion-cache-1y4p8pa {width: 100%; padding: 6rem 1rem 10rem; max-width: 65rem;} 
        .st-emotion-cache-7ym5gk {padding: 0.0rem 0.0rem; min-height: 0px;} 
        .st-emotion-cache-ue6h4q {min-height: 0rem;} 
.st-emotion-cache-1y4p8pa {width: 100%; padding: 3rem 1rem 1rem; max-width: 65rem;}
[data-testid='stFileUploader'] { display: flex;width: max-content;align-items: center; } [data-testid='stFileUploader'] section { padding: 0; float: left; } [data-testid='stFileUploader'] section > input + div { display: none; } [data-testid='stFileUploader'] section + div { float: right; padding-top: 0; }
[data-testid='stFileUploader'] label { margin-right: 10px; /* Adjust spacing between label and button */ }
</style>'''
st.markdown("###### DNA Leaf Models (QC)")

###############################################################################################
r11, r12,r13 = st.columns([2,1,2])
uploaded_file = r11.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
st.markdown(css, unsafe_allow_html=True)
leafimageprocessor = LeafImageProcessor()

if "Process" not in st.session_state:
    st.session_state.Process = False

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    o_width, o_height = original_image.size
    # if o_width!=o_height:
    original_image = leafimageprocessor.imageBanding_new(original_image)
    dominant_color = leafimageprocessor.get_background_color(original_image)
    resized_original_image = leafimageprocessor.resize_image_for_display(original_image)
    # col1, col2 = st.columns([1, 4])
    # col1.write("Select a value:")
    # col2.slider("", min_value=0, max_value=100, value=50)
    r12.write("Select Rotation Angle (in degrees)")
    angle = r13.slider("", -180, 180, 5)
    rotated_image = leafimageprocessor.rotate_image(original_image, angle, dominant_color)

    preprocessed_image = leafimageprocessor.preprocess_image(rotated_image)
    border_image = leafimageprocessor.extract_leaf_border(preprocessed_image)

    apex, base = leafimageprocessor.split_horizontally(rotated_image)
    apex_left, apex_right = leafimageprocessor.split_vertically(rotated_image)
    # base_left, base_right = leafimageprocessor.split_vertically(base)

    apex_border, base_border, left_border, right_border = leafimageprocessor.split_leaf_border(border_image)
    st.markdown("###### Original, Rotated, and Border Images")
    cols = st.columns(11)
    cols[0].image(resized_original_image, caption="Original Image",width=110)
    cols[1].image(rotated_image, caption=f"Rotated ({angle}°)",width=110)
    cols[2].image(border_image, caption="Border Image",width=110)

    # col1, col2, col3,col4, col5, col6 = st.columns(11)
    # col1.image(resized_original_image, caption="Original Image")
    # col2.image(rotated_image, caption=f"Rotated ({angle}°)")
    # col3.image(border_image, caption="Border Image")

    # Split Image Display
    # st.markdown("###### Split Images")
    # cols = st.columns(8)
    cols[3].image(apex, caption="Apex",width=110)
    cols[4].image(base, caption="Base",width=110)
    # cols[2].image(base_left, caption="Base Left",width=200)
    # cols[3].image(base_right, caption="Base Right",width=200)
    cols[5].image(apex_left, caption="Left",width=110)
    cols[6].image(apex_right, caption="Right",width=110)

    # st.markdown("###### Border Splits")
    # border_cols = st.columns(6)
    cols[7].image(apex_border, caption="Apex Border",width=110)
    cols[8].image(base_border, caption="Base Border",width=110)
    cols[9].image(left_border, caption="Left Border",width=110)
    cols[10].image(right_border, caption="Right Border",width=110)

###################################### MODELS CALLING ############################################
model_paths = {     
            r"C:\DNA\codes\ModelCODE\Drawingscanned_shape_20241218.keras": r"W:\ModelInput\V2\drawing_shape_old",
            r"C:\DNA\codes\ModelCODE\Drawingscanned_base_20241210.keras": r"W:\ModelInput\V2\drawing_base",
            r"C:\DNA\codes\ModelCODE\Drawingscanned_apex_20241210.keras": r"W:\ModelInput\V2\drawing_apex",
            r"C:\DNA\codes\ModelCODE\Drawingscanned_margin_20241210.keras": r"W:\ModelInput\V2\drawing_margin"
        }
if uploaded_file is not None:
    if st.button("Process"):
        st.session_state.Process = True

if st.session_state.Process:        
    DNAmodels = DNA_Models(model_paths)
    st.markdown("###### Model Results")       

    img1=Image.open(uploaded_file)
    # img_array1 = np.array(img1)
    
    ## Shape ##
    shape_banding_image = leafimageprocessor.imageBanding_new(img1)
    shape_image = leafimageprocessor.preprocess_for_model(border_image)
    predicted_class, confidence, image_id, image_path = DNAmodels.predict_and_find_highest_confidence("Shape", shape_image)

    img = cv2.imread(image_path)
    shape_cols = st.columns(8)
    shape_cols[0].image(shape_image, caption="Shape_"+f"{predicted_class}")
    shape_cols[1].image(img, caption=f"{confidence:.2f}")

    ## Apex ##
    apex_banding_image = leafimageprocessor.imageBanding(apex_border)
    apex_image = leafimageprocessor.preprocess_for_model(apex_banding_image)
    predicted_class, confidence, image_id, image_path = DNAmodels.predict_and_find_highest_confidence("Apex", apex_image)
    img = cv2.imread(image_path)
    shape_cols[2].image(apex_image, caption="Apex_"+ f"{predicted_class}")
    shape_cols[3].image(img, caption=f" {confidence}")

    ## Base ##
    base_banding_image = leafimageprocessor.imageBanding(base_border)
    base_image = leafimageprocessor.preprocess_for_model(base_banding_image)
    predicted_class, confidence, image_id, image_path = DNAmodels.predict_and_find_highest_confidence("Base", base_image)
    img = cv2.imread(image_path)
    shape_cols[4].image(base_image, caption="Base_"+f"{predicted_class}")
    shape_cols[5].image(img, caption=f"{confidence:.2f}")

    ## Margin ##
    margin_banding_image = leafimageprocessor.imageBanding(border_image)
    margin_image = leafimageprocessor.preprocess_for_model(margin_banding_image)
    predicted_class, confidence, image_id, image_path = DNAmodels.predict_and_find_highest_confidence("Margin", margin_image)
    img = cv2.imread(image_path)
    shape_cols[6].image(border_image, caption="Margin_"+f"{predicted_class}")
    shape_cols[7].image(img, caption=f"{confidence:.2f}")

    ## Shape ##
    shape_banding_image = leafimageprocessor.imageBanding(original_image)
    shape_image = leafimageprocessor.preprocess_for_model(border_image)
    predicted_class, confidence, image_id, image_path = DNAmodels.predict_and_find_highest_confidence("Shape", shape_image)

    img = cv2.imread(image_path)
    shape_row = st.columns(8)
    shape_row[0].image(original_image, caption="Shape_"+f"{predicted_class}")
    shape_row[1].image(img, caption=f"{confidence:.2f}")

    ## Apex ##
    apex_banding_image = leafimageprocessor.imageBanding(apex)
    apex_image = leafimageprocessor.preprocess_for_model(apex)
    predicted_class, confidence, image_id, image_path = DNAmodels.predict_and_find_highest_confidence("Apex", apex_image)
    img = cv2.imread(image_path)
    shape_row[2].image(apex, caption="Apex_"+ f"{predicted_class}")
    shape_row[3].image(img, caption=f" {confidence}")

    ## Base ##
    base_banding_image = leafimageprocessor.imageBanding(base)
    base_image = leafimageprocessor.preprocess_for_model(base)
    predicted_class, confidence, image_id, image_path = DNAmodels.predict_and_find_highest_confidence("Base", base_image)
    img = cv2.imread(image_path)
    shape_row[4].image(base, caption="Base_"+f"{predicted_class}")
    shape_row[5].image(img, caption=f"{confidence:.2f}")

    ## Margin ##
    margin_banding_image = leafimageprocessor.imageBanding(original_image)
    margin_image = leafimageprocessor.preprocess_for_model(original_image)
    predicted_class, confidence, image_id, image_path = DNAmodels.predict_and_find_highest_confidence("Margin", margin_image)
    img = cv2.imread(image_path)
    shape_row[6].image(original_image, caption="Margin_"+f"{predicted_class}")
    shape_row[7].image(img, caption=f"{confidence:.2f}")

if uploaded_file is not None:
    ### Save option ###
    save_dir = "shape_saved_files"
    file_extension = os.path.splitext(uploaded_file.name)[1]
    file_name = os.path.splitext(uploaded_file.name)[0]
    col1, col2, col22  = st.columns([1,2,1])
    with col1:
        shape_options = ["--Select Shape--","acicular","bilobed","cordate","cuneate","deltoid",
        "elliptic","flabellate","hastate","lanceolate","linear","lyrate","obcordate","obdeltoid",
        "oblanceolate","oblong","obovate","orbicular","orbiculate","ovate","peltate","reniform",
        "rhombic","rhomboid","sagittate""segittate","spathulate"]
        
        selected_shape_option = st.selectbox("Select Shape",shape_options,key="selected")

    with col2:
        if st.button("Save Shape Image") and uploaded_file is not None:
            if selected_shape_option != "--Select Shape--":
                shape_save_path = os.path.join(save_dir, f"SHAPE_{file_name}_{selected_shape_option.replace(' ', '_')}{file_extension}")
                leafimageprocessor.saveImageFile(uploaded_file,shape_save_path)
    with col22:
        st.markdown(
            """
                <a href="http://localhost/DNAFiles/Shape.html" target="_blank">Shape info</a>
            """,
            unsafe_allow_html=True
        )
    col3, col4, col44  = st.columns([1,2,1])
    with col3:
        base_options=["--Select Base--"," assymetrical","attenuate","auriculate","cordate",
        "cuneate","hastate","oblique","obtuse","rounded","sagittate","subcordate","truncate"]

        selected_base_option = st.selectbox("Select Base", base_options)
    with col4:
        if st.button("Save Base Image") and uploaded_file is not None:
            if selected_shape_option != "--Select Base--":
                base_save_path = os.path.join(save_dir, f"BASE_{file_name}_{selected_base_option.replace(' ', '_')}{file_extension}")
                leafimageprocessor.saveImageFile(uploaded_file,base_save_path)
    with col44:
        st.markdown(""" <a href="http://localhost/DNAFiles/Base.html" target="_blank">Base info</a>
            """, unsafe_allow_html=True)

    col5, col6, col66  = st.columns([1, 2,1])
    with col5:
        apex_options=["--Select Apex--",
        "acuminate","acute","apiculate","aristate","caudate","cirrhose",
        "cleft","cuspidate","emarginate","mucronate","mucronulate","obcordate",
        "obtuse","retuse","rounded","spinose","truncate"]
        
        selected_apex_option = st.selectbox("Select Apex", apex_options)
    with col6:
        if st.button("Save Apex Image") and uploaded_file is not None:
            if selected_apex_option != "--Select Apex--":
                apex_save_path = os.path.join(save_dir, f"APEX_{file_name}_{selected_apex_option.replace(' ', '_')}{file_extension}")
                leafimageprocessor.saveImageFile(uploaded_file,apex_save_path)
    with col66:
        st.markdown("""<a href="http://localhost/DNAFiles/Apex.html" target="_blank">Apex info</a>
            """,unsafe_allow_html=True)
        
    col7, col8, col88  = st.columns([1, 2,1])
    with col7:
        margin_options=["--Select Margin--","crenate","crenulate","dentate","denticulate",
        "entire","incised","retrorse","serrate","serrulate","sinuate-undulate"]
        selected_margin_option = st.selectbox("Select Margin", margin_options)
    with col8:
        if st.button("Save Margin Image") and uploaded_file is not None:
            if selected_margin_option != "--Select Margin--":
                margin_save_path = os.path.join(save_dir, f"MARGIN_{file_name}_{selected_margin_option.replace(' ', '_')}{file_extension}")
                st.write(leafimageprocessor.saveImageFile(uploaded_file,margin_save_path))
    with col88:
        st.markdown("""<a href="http://localhost/DNAFiles/Margin.html" target="_blank">Margin info</a>
            """, unsafe_allow_html=True)


        





