import streamlit as st
import numpy as np
import torch
import random
import os
#from your_module import Net, convert_to_midi  # assuming you have your functions in a separate module
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
import ui_01
from Models import Wavenet,LSTM
Net = torch.load('Trained_Model/model.pth')

# Define a function to generate music

#######Convert midi to 
# from midi2audio import FluidSynth
# fs = FluidSynth()
# fs.midi_to_audio('music.mid', 'music.wav')

#st.audio("music.mid", format="audio/mid")

# UI
# Display the larger title using HTML
st.markdown("<h1 style='text-align: center;'>DKTE Society's</h1>", unsafe_allow_html=True)
st.title("Textile and Engineering Institute, Ichalkaranji")
st.header("Crafting Music with Recurrent Neural Networks")


st.text("How many Seed value mixture :")
n=st.number_input("Insert the value :",min_value=1, max_value=20, step=1)

def generate_music():
    for j in range(n):
        index = random.randint(0, len(ui_01.x_test))
        st.write("Taking the seed tune as follows:")
        st.text_area("Value:",ui_01.x_test[index])
        tune = ui_01.x_test[index]
        input_data = np.empty((1, 32), dtype=int)
        input_data[0] = tune
        input_data = torch.from_numpy(input_data)
        next_preds = 64
        for i in range(next_preds):
            output = Net(input_data.to(device), input_data.shape[0])
            next_preds = np.argmax(output.cpu().detach().numpy(), axis=1)
            input_data = input_data.cpu().detach().numpy()
            input_data = torch.from_numpy(np.array([np.append(j, next_preds[ind]) 
                                               for ind, j in enumerate(input_data)])[:, 1:])
            tune = np.insert(tune, -1, next_preds[0])
        tune = [ui_01.unique_notes[i] for i in tune]
        path = './Outputs/music'+str(j)+'.midi'
        ui_01.midi_helper.convert_to_midi(tune)  # Remove the path argument


# Button to generate music
if st.button("Generate Music"):
    generate_music()
    st.success("Music generated successfully!")
    
    
    # current_directory = os.getcwd()
    # midi_files = "music.mid"
    # midi_file_path = os.path.join(current_directory, midi_files)

    # st.text("Full path to the MIDI file: " + midi_file_path)

    
    # from datetime import datetime
    # current_directory = os.getcwd()
    # midi_file_name = "music.mid"
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp to the file name
    # new_midi_file_name = f"music_{timestamp}.mid"  # New unique file name
    # midi_file_path = os.path.join(current_directory, new_midi_file_name)

    #st.markdown(f"Full path to the MIDI file: <a href='file://{midi_file_path}' target='_blank'>{midi_file_path}</a>", unsafe_allow_html=True)


#     import os
# #old_filename = ('C:/Users/Admin/Python/Music_generation/Music-Gen-AI-main/music.mid')
# new_filename = st.text_input("Enter new filename:",key=3)
# #os.rename(old_filename, new_filename)

    # if st.button("Rename File"):
    #     if new_filename.strip():
    #         try:
    #             os.rename(old_filename, new_filename)
    #             st.success("File renamed successfully!")
    #         except Exception as e:
    #             st.error(f"An error occurred: {e}")
    #     else:
    #         st.warning("Please provide a new filename.")
    #st.audio("music.mid")
